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
    return int(np.log2(x))


def stage_of_resolution(res: int) -> int:
    """Measure of the stage of growth of the model.
    Also corresponds to the number of intermediate discriminator blocks. 
    Stage 0: 4x4 resolution
    Stage 1: 8x8 input resolution, etc.
    """
    return log2(res) - 2

def is_valid_resolution(res: int) -> bool:
    res_log2 = log2(res)
    return 2**res_log2 == res and res >= 4

def assert_valid_resolution(res: int) -> None:
    if not is_valid_resolution(res):
        raise RuntimeError(f"Invalid resolution: {res} (must be a power of 2 no less than 4)")

def resolution_of_stage(stage: int) -> int:
    return 2 ** (stage+2)


def filters_for(resolution) -> int:
    res_log2 = log2(resolution)
    return num_filters(res_log2-1)


def num_filters(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    """Gives the number of feature maps that is reasonable for a given stage.
    Arguments:
        fmap_base: Overall multiplier for the number of feature maps.        
        fmap_decay: log2 feature map reduction when doubling the resolution.        
        fmap_max: Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)




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
