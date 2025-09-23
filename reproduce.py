from time_trans_aae.aae import aae_model
from tools import MinMaxScaler
from evaluation_model import disc_eva, fore_eva
from time_trans_aae.networks import timesformer_dec, cnn_enc, cnn_dec, cautrans_dec, discriminator
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.model_selection import train_test_split
import scipy.stats


dataset = 'sine_cpx'            #use Sine_cpx as an example
valid_perc = 0.1
train_perc = 1-valid_perc
full_train_data = np.load('datasets/'+dataset+'.npy')
N, T, D = full_train_data.shape

N_train = int(N * (1 - valid_perc))
N_valid = N - N_train
np.random.shuffle(full_train_data)
train_data = full_train_data[:N_train]
valid_data = full_train_data[N_train:]
scaler = MinMaxScaler()
x_train = scaler.fit_transform(train_data)
x_valid = scaler.transform(valid_data)

fig, axs = plt.subplots(5, 1, figsize=(6,5), sharex=True)
for i in range(5):
    rnd_idx = np.random.choice(len(x_train))
    s = x_train[rnd_idx]
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].plot(s)
st = plt.suptitle('Original')
st.set_y(0.93)


ts_shape = x_train.shape[1:]
latent = 16

enc = cnn_enc(
    input_shape=ts_shape,
    latent_dim=latent,
    n_filters=[64, 128, 256],
    k_size=4,
    dropout=0.2
)

dec = timesformer_dec(
    input_shape=(latent,),
    ts_shape=ts_shape,
    head_size=64,
    num_heads=3,
    n_filters=[128, 64],
    k_size=4,
    dilations=[1,4],
    dropout=0.2
)

disc = discriminator(input_shape=(latent,), hidden_unit=32)

def ae_loss(ori_ts, rec_ts):
    return tf.keras.metrics.mse(ori_ts, rec_ts)

def dis_loss(y_true, y_pred):
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

def gen_loss(y_true, y_pred):
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)


ae_schedule = PolynomialDecay(initial_learning_rate=0.005, decay_steps=600, end_learning_rate=0.0001, power=0.5)
dc_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)
ge_schedule = PolynomialDecay(initial_learning_rate=0.001, decay_steps=300, end_learning_rate=0.0001, power=0.5)


ae_opt = tf.keras.optimizers.Adam(ae_schedule)
dc_opt = tf.keras.optimizers.Adam(dc_schedule)
ge_opt = tf.keras.optimizers.Adam(ge_schedule)


model = aae_model(
    encoder=enc,
    decoder=dec,
    discriminator=disc,
    latent_dim=latent,
    dis_steps=1,
    gen_steps=1)


model.compile(rec_opt=ae_opt, rec_obj=ae_loss, dis_opt=dc_opt, dis_obj=dis_loss, gen_opt=ge_opt, gen_obj=gen_loss)

history = model.fit(x_train, epochs=800, batch_size=230)


z = tf.random.normal([x_train.shape[0], latent], 0.0, 1.0)
sample = model.dec.predict(z)
# sample = np.clip(sample, 0, 1)

train_ori = np.mean(x_train, axis=-1)
train_gen = np.mean(sample, axis=-1)

select = x_train.shape[0]
idx = np.random.permutation(select)
ori = train_ori[idx]
gen = train_gen[idx]
prep_data_final = np.concatenate((ori, gen), axis = 0)

emb = TSNE(n_components=2, learning_rate=10, init = 'random', perplexity=50, max_iter=400).fit_transform(prep_data_final)
plt.yticks([])
plt.xticks([])
plt.scatter(emb[:select, 0], emb[:select, 1], alpha=0.2, color='b')
plt.scatter(emb[select:, 0], emb[select:, 1], alpha=0.2, color='r')

plt.savefig('tsne.png')


res_d = []
for _ in range(5):  # run 5 times to calculate avg and CI
    x_train_d = np.concatenate((x_train, sample))
    y_train_d = np.append(np.ones(x_train.shape[0]), np.zeros(sample.shape[0]))
    disc_m = disc_eva(input_shape=x_train_d.shape[1:], rnn_unit=[128], dropout=0.3)
    disc_m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    disc_m.fit(x_train_d, y_train_d, epochs=50, batch_size=1000, verbose=1)

    test_gen = model.dec.predict(tf.random.normal([N_valid, latent], 0.0, 1.0))
    test_ori = x_valid
    x_test_d = np.concatenate((test_ori, test_gen))
    y_test_d = np.append(np.ones(test_ori.shape[0]), np.zeros(test_gen.shape[0]))
    l, acc = disc_m.evaluate(x_test_d, y_test_d)
    res_d.append(np.abs(acc-0.5))
