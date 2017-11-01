import os
import sys
import time
import models as models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
# Utils
sys.path.append("../utils")
import data_utils

def train(cat_dim,noise_dim,batch_size,n_batch_per_epoch,nb_epoch,dset="mnist"):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """
    # Load and rescale data
    if dset == "mnist":
        X_real_train, _, _, _ = data_utils.load_mnist()

    img_dim = X_real_train.shape[-3:]
    epoch_size = n_batch_per_epoch * batch_size

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-4, momentum=0.9, nesterov=True)

        # Load generator model
        generator_model = models.load("generator_deconv", cat_dim, noise_dim, img_dim, batch_size, dset=dset)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator", cat_dim, noise_dim, img_dim, batch_size, dset=dset)

        generator_model.compile(loss='mse', optimizer=opt_discriminator)
        # stop the discriminator to learn while in generator is learning
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model, discriminator_model, cat_dim, noise_dim)

        list_losses = ['binary_crossentropy', 'categorical_crossentropy']
        list_weights = [1, 1]
        DCGAN_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_dcgan)

        # Multiple discriminator losses
        # allow the discriminator to learn again
        discriminator_model.trainable = True
        discriminator_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_discriminator)
        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_real_batch in data_utils.gen_batch(X_real_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc, y_cat = data_utils.get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, cat_dim, noise_dim)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, [y_disc, y_cat])

                # Create a batch to feed the generator model
                X_noise, y_gen, X_cat = data_utils.get_gen_batch(batch_size, cat_dim, noise_dim)

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch([X_cat, X_noise], [y_gen, X_cat])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                progbar.add(batch_size, values=[("D tot", disc_loss[0]),
                                                ("D log", disc_loss[1]),
                                                ("D cat", disc_loss[2]),
                                                ("G tot", gen_loss[0]),
                                                ("G log", gen_loss[1]),
                                                ("G cat", gen_loss[2])])

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    data_utils.plot_generated_batch(X_real_batch, generator_model, batch_size, cat_dim, noise_dim)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            if e % 5 == 0:
                gen_weights_path = os.path.join('../../models/gen_weights_epoch%s.h5' % (e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/disc_weights_epoch%s.h5' % (e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/DCGAN_weights_epoch%s.h5' % (e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
