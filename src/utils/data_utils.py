from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py

import matplotlib.pylab as plt


def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.


def load_mnist():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train = normalization(X_train)
    X_test = normalization(X_test)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def gen_batch(X, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx]

def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def sample_cat(batch_size, cat_dim):

    y = np.zeros((batch_size, cat_dim[0]), dtype="float32")
    random_y = np.random.randint(0, cat_dim[0], size=batch_size)
    y[np.arange(batch_size), random_y] = 1
    return y

def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, cat_dim, noise_dim, noise_scale=0.5,label_smoothing=True):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Pass noise to the generator
        y_cat = sample_cat(batch_size, cat_dim)
        noise_input = sample_noise(noise_scale, batch_size, noise_dim)
        # Produce an output
        X_disc = generator_model.predict([y_cat, noise_input],batch_size=batch_size)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        # y_disc[:, 0] = 1
    else:
        X_disc = X_real_batch
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        y_cat = sample_cat(batch_size, cat_dim)
        if label_smoothing:
            y_disc[:, 0] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 0] = 1

    return X_disc, y_disc, y_cat

def get_gen_batch(batch_size, cat_dim, noise_dim, noise_scale=0.5):

    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((X_gen.shape[0], 1), dtype=np.uint8)
    # y_gen[:, 0] = 1

    y_cat = sample_cat(batch_size, cat_dim)

    return X_gen, y_gen, y_cat

def plot_generated_batch(X_real, generator_model, batch_size, cat_dim, noise_dim, noise_scale=0.5):

    # Generate images
    y_cat = sample_cat(batch_size, cat_dim)
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_gen = generator_model.predict([y_cat, noise_input],batch_size=batch_size)

    X_real = inverse_normalization(X_real)
    X_gen = inverse_normalization(X_gen)

    Xg = X_gen[:8]
    Xr = X_real[:8]
    X = np.concatenate((Xg, Xr), axis=0)
    list_rows = []
    for i in range(int(X.shape[0] / 4)):
        Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
        list_rows.append(Xr)

    Xr = np.concatenate(list_rows, axis=0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig("../../figures/current_batch.png")
    plt.clf()
    plt.close()