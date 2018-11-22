from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
from models import *
import sys
import os
import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 25

        optimizer = Adam(0.0001, 0.5)

        try:
            self.discriminator = load_model('testD.h5')
            self.generator = load_model('testG.h5')
        except:
            self.discriminator = discriminator128()
            self.discriminator.compile(loss='binary_crossentropy',
            metrics=['accuracy'],optimizer=optimizer)

            self.generator = generator128()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.summary()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = np.load('celeb128-5000.npy')
        X_train = (X_train.astype(np.float32) / 255)
        
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.generator.save('testG.h5')
                self.discriminator.save('testD.h5')

    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 0.4, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("im/faces_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=1000000, batch_size=32, save_interval=50)
