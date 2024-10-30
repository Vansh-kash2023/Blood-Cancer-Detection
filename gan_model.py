import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class GAN:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.latent_dim = 100  # Dimension of the noise vector

        self.optimizer = Adam(0.0002, 0.5)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.combined = self.build_combined()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(7 * 7 * 256, use_bias=False, input_dim=self.latent_dim))
        model.add(BatchNormalization())  # Normalize activations for stability
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def build_combined(self):
        self.discriminator.trainable = False  # Freeze discriminator during generator training
        model = Sequential([self.generator, self.discriminator])
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model

    def train(self, train_dataset, epochs=1000, batch_size=32):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            # Train Discriminator
            for real_images, _ in train_dataset.take(len(train_dataset) // batch_size):
                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                fake_images = self.generator.predict(noise)

                d_loss_real = self.discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Save generated images every 10 epochs
            if epoch % 10 == 0:
                self.save_generated_images(epoch)

            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    def save_generated_images(self, epoch):
        noise = np.random.normal(0, 1, (25, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

        os.makedirs("gan_images", exist_ok=True)

        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.savefig(f"gan_images/generated_epoch_{epoch}.png")
        plt.close()

# Usage example:
if __name__ == '__main__':
    # Load your dataset here
    img_height, img_width = 256, 256  # Adjust according to your image size
    img_shape = (img_height, img_width, 1)  # Shape of the input images (grayscale)

    # Load and preprocess data (adjust paths and preprocessing as needed)
    import pathlib
    train_dir = ("data/normal/healthy")
    train_dataset = image_dataset_from_directory(
        train_dir,
        batch_size=32,
        color_mode='grayscale',
        shuffle=True
    )

    # Optionally, resize images if needed:
    def resize_image(image, label):
        image = tf.image.resize(image, (img_height, img_width))
        return image, label

    train_dataset = train_dataset.map(resize_image)

    gan = GAN(img_shape)
    gan.train(train_dataset, epochs=1000, batch_size=32)