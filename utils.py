import tensorflow as tf
import numpy as np

def NHWC_to_NCHW(images: tf.Tensor) -> tf.Tensor:
    return tf.transpose(images, [0, 3, 1, 2])


def NCHW_to_NHWC(images: tf.Tensor) -> tf.Tensor:
    return tf.transpose(images, [0, 2, 3, 1])


def normalize_images(images: tf.Tensor) -> tf.Tensor:
    return (tf.cast(images, tf.float32) - 127.5) / 127.5


def unnormalize_images(images: tf.Tensor) -> tf.Tensor:
    return tf.cast(images * 127.5 + 127.5, tf.uint8)

def log2(x: int) -> int:
    x_log2 = int(np.log2(x))
    if 2**x_log2 != x:
        raise RuntimeError(f"Resolution {x} is not a power of two.")
    return x_log2


class HeInitializer(tf.keras.initializers.VarianceScaling):
    """Simple Class-implementation of the He Normal Initializer"""

    def __init__(self, gain=2.0):
        super().__init__(scale=gain)

    @classmethod
    def get_constant(cls, shape: tf.TensorShape, gain: float = 2, fan_in: int = None) -> tf.Tensor:
        """Computes the `He initializer` scaling constant with respect to the given kernel shape.

        Arguments:
            shape: The shape of the kernel or weight to be used.
            gain: The gain to be used as the numerator in the calculation of the constant.
            fan_in: optional override of the number of parameters. If omitted, the product of the values of shape[1:] is used. 

        Returns:
            A tf.constant Tensor containing the scaling constant.
        """
        if fan_in is None:
            fan_in = np.prod([d.value for d in shape[:-1]])
        # He init scaling factor
        std = np.sqrt(gain / fan_in)
        return tf.constant(std, dtype=float, name="wscale")
