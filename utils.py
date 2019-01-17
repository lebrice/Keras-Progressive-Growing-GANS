import tensorflow as tf


def NHWC_to_NCHW(images: tf.Tensor) -> tf.Tensor:
    return tf.transpose(images, [0, 3, 1, 2])


def NCHW_to_NHWC(images: tf.Tensor) -> tf.Tensor:
    return tf.transpose(images, [0, 2, 3, 1])


def normalize_images(images: tf.Tensor) -> tf.Tensor:
    return (tf.cast(images, tf.float32) - 127.5) / 127.5


def unnormalize_images(images: tf.Tensor) -> tf.Tensor:
    return tf.cast(images * 127.5 + 127.5, tf.uint8)
