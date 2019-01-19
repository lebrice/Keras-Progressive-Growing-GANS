"""
Training script.
Taken in big part from this DCGAN tutorial:
https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
"""
import time
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.enable_eager_execution()

from itertools import islice
from progressbar import progressbar


from utils import NHWC_to_NCHW, normalize_images
from math import ceil
import os

from networks_keras import Downscale2D
from discriminator import Discriminator
from generator import Generator

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss



cifar10 = tf.keras.datasets.cifar10
(train_images, _), (test_images, _) = cifar10.load_data()

train_images = normalize_images(train_images)
train_images = NHWC_to_NCHW(train_images)

test_images = normalize_images(test_images)
test_images = NHWC_to_NCHW(test_images)

train_images_32x32 = train_images
train_images_16x16 = Downscale2D()(train_images_32x32)
train_images_8x8 = Downscale2D()(train_images_16x16)
train_images_4x4 = Downscale2D()(train_images_8x8)

test_images_32x32 = test_images
test_images_16x16 = Downscale2D()(test_images_32x32)
test_images_8x8 =   Downscale2D()(test_images_16x16)
test_images_4x4 =   Downscale2D()(test_images_8x8)

BATCH_SIZE = 32
BUFFER_SIZE = 1000

train_dataset_32x32 = (tf.data.Dataset.from_tensor_slices(train_images_32x32)
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE))
train_dataset_16x16 = (tf.data.Dataset.from_tensor_slices(train_images_16x16)
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE))
train_dataset_8x8 = (tf.data.Dataset.from_tensor_slices(train_images_8x8)
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE))
train_dataset_4x4 = (tf.data.Dataset.from_tensor_slices(train_images_4x4)
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE))

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.contrib.eager.defun
def train_step(images):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])
    # print(images)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.variables))

noise_dim = 512
num_examples_to_generate = 16

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate, noise_dim])


def train_eager(dataset, epochs):
    # TODO: find this for real.
    steps_per_epoch = ceil(50_000 / BATCH_SIZE)
    steps_per_epoch = 100

    
    for epoch in range(epochs):
        start = time.time()
        for i, images in enumerate(islice(dataset, steps_per_epoch)):
            train_step(images)
            print(f"train_step {i}/{steps_per_epoch} done")
        
        # generate_and_save_images(generator, epoch, random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time taken for epoch {epoch + 1} is {time.time()-start:.1f} sec")
    


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    images = model(test_input, training=False)
    from utils import NCHW_to_NHWC, unnormalize_images
    images = NCHW_to_NHWC(images)
    images = unnormalize_images(images)

    fig = plt.figure(figsize=(4, 4))

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

EPOCHS = 1
train_eager(train_dataset_4x4, EPOCHS)
generator.grow()
train_eager(train_dataset_8x8, EPOCHS)
generator.grow()
train_eager(train_dataset_16x16, EPOCHS)
generator.grow()
train_eager(train_dataset_32x32, EPOCHS)
