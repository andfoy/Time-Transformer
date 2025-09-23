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
from tensorflow.keras.activations import relu
from sklearn.model_selection import train_test_split
import scipy.stats
import polars as pl


df = pl.read_csv("datasets/Greenhouse_Control_final_28.csv", infer_schema_length=0)
df_floats = df.with_columns(pl.all().cast(pl.Float64, strict=False))

non_null = df_floats.select(pl.count('*'))
total_per_column = df_floats.select(pl.all().len())
diff = total_per_column - non_null

null_cols = [col.name for col in diff if (col == 71).item()]
null_cols
proj = df_floats.select(null_cols)

idx = proj.select(pl.col("*").is_null().arg_true())

filtered = df_floats.with_row_index().filter(~pl.col("index").is_in(idx['AssimLight']))
filtered

fnon_null = filtered.select(pl.count('*'))
fnon_null
ftotal_per_column = filtered.select(pl.all().len())
fdiff = ftotal_per_column - fnon_null
fdiff

null_cols = [col.name for col in fdiff if (col != 0).item()]

fproj = filtered.select(null_cols)
fproj
idx = fproj.select(pl.col("*").is_null().arg_true())
idx

filtered.with_columns(index = pl.int_range(pl.len()))
dataset = filtered.with_columns(
   co2_dos = pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise("co2_dos"))

in_data = dataset.to_numpy()[:, 2:]
# N = 1
# T, D = in_data.shape

full_train_data = in_data[:-2]
full_train_data = full_train_data.reshape(663, 72, 28)
N, T, D = full_train_data.shape


# dataset = 'sine_cpx'            #use Sine_cpx as an example
valid_perc = 0.1
train_perc = 1 - valid_perc
# full_train_data = np.load('datasets/'+dataset+'.npy')
# N, T, D = full_train_data.shape

N_train = int(N * (1 - valid_perc))
N_valid = N - N_train
np.random.shuffle(full_train_data)
train_data = full_train_data[:N_train]
valid_data = full_train_data[N_train:]
scaler = MinMaxScaler()
x_train = scaler.fit_transform(train_data)
x_valid = scaler.transform(valid_data)

# fig, axs = plt.subplots(5, 1, figsize=(6,5), sharex=True)
# for i in range(5):
#     rnd_idx = np.random.choice(len(x_train))
#     s = x_train[rnd_idx]
#     axs[i].set_yticks([])
#     axs[i].set_xticks([])
#     axs[i].plot(s)
# st = plt.suptitle('Original')
# st.set_y(0.93)


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

def range_loss(rec_ts):
    return 3 * tf.math.reduce_mean(relu(-rec_ts), -1) + 3 * tf.math.reduce_mean(relu(rec_ts - 1), -1)

def ae_loss(ori_ts, rec_ts):
    return tf.keras.metrics.mse(ori_ts, rec_ts)

def dis_loss(y_true, y_pred):
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

def gen_loss(y_true, y_pred):
    return tf.keras.metrics.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)


ae_schedule = PolynomialDecay(initial_learning_rate=0.005, decay_steps=300, end_learning_rate=0.0001, power=0.5)
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


model.compile(rec_opt=ae_opt, rec_obj=ae_loss, dis_opt=dc_opt, dis_obj=dis_loss, gen_opt=ge_opt, gen_obj=gen_loss, range_loss=range_loss)

history = model.fit(x_train, epochs=2000, batch_size=663)
