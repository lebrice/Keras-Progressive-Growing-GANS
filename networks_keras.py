import tensorflow as tf
import numpy as np


# tf.enable_eager_execution()

from typing import NamedTuple, Type
from networks import get_weight
from typing import List, Tuple, Iterable
from functools import wraps

DATA_FORMAT = "channels_first"


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


def WeightScaled(
        layer_base_class: Type[tf.keras.layers.Layer],
        kernel_attr: str = "kernel",
        kernel_initializer_attr: str = "kernel_initializer",
):
    """Wraps a given Keras layer base-class, adding Runtime Weight Scaling in the `build()` method using the constant from the He initializer.

    Arguments:
        `kernel_attr`:
            The name of the kernel attribute (or equivalent) within the base class, to be scaled.
        `kernel_initializer_attr`: 
            The name of the kernel initializer attribute within the base class.
            Will be replace by a Random Normal Variable with mean 0 and standard deviation of 1.  
    """
    class WeightScaledVariant(layer_base_class):
        def __init__(self, gain: float = 2.0, fan_in: int = None, *args, **kwargs):
            self.gain = gain
            self.fan_in = fan_in
            kwargs.update(
                {kernel_initializer_attr: tf.keras.initializers.RandomNormal(
                    0, 1)}
            )
            super().__init__(
                *args,
                **kwargs
            )

        def build(self, input_shape):
            super().build(input_shape)
            kernel: tf.Tensor = getattr(self, kernel_attr)
            # , gain=self.gain, fan_in=self.fan_in)
            scaling_constant = HeInitializer.get_constant(kernel.shape)
            kernel = kernel * scaling_constant
            setattr(self, kernel_attr, kernel)
    return WeightScaledVariant


class WeightScaledDense(WeightScaled(tf.keras.layers.Dense)):
    def __init__(self,
        units: int,
        activation=tf.nn.leaky_relu,
        use_bias=True,
        *args, **kwargs):
        super().__init__(units=units, activation=activation, use_bias=use_bias, *args, **kwargs)


class WeightScaledConv2D(WeightScaled(tf.keras.layers.Conv2D)):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 activation=tf.nn.leaky_relu,
                 data_format="channels_first",
                 padding="SAME",
                 use_bias=True,
                 *args, **kwargs):
        super().__init__(
            filters=filters, kernel_size=kernel_size, activation=activation,
            data_format=data_format, padding=padding, *args, **kwargs)



class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

# ----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.


# def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
#     assert kernel >= 1 and kernel % 2 == 1
#     w = get_weight([kernel, kernel, x.shape[1].value, fmaps],
#                    gain=gain, use_wscale=use_wscale)
#     w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
#     w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
#     w = tf.cast(w, x.dtype)
#     return tf.nn.conv2d(x, w, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')


class Upscale2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, *args, **kwargs):
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor):
        if self.factor == 1:
            return x
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, self.factor, 1, self.factor])
        x = tf.reshape(x, [-1, s[1], s[2] * self.factor, s[3] * self.factor])
        return x


class ToRGB(WeightScaledConv2D):
    def __init__(self, num_channels: int = 3, *args, **kwargs):
        super().__init__(
            gain=1,
            filters=num_channels,
            kernel_size=1,
            *args,
            **kwargs,
        )


class FirstGeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, normalize_latents: bool = True, pixelnorm_epsilon=1e-8, *args, **kwargs):
        self.normalize_latents = normalize_latents
        self.pixelnorm = PixelNorm(pixelnorm_epsilon)
        self.dense = WeightScaledDense(
            units=512 * 4 * 4,
        )
        self.reshape = tf.keras.layers.Reshape([512, 4, 4])
        self.conv = WeightScaledConv2D(
            filters=512,
            kernel_size=3,
        )
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor):
        if self.normalize_latents:
            x = self.pixelnorm(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.pixelnorm(x)
        x = self.conv(x)
        x = self.pixelnorm(x)
        return x


class GeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, res: int, *args, **kwargs):
        self.res = res
        self.upscale2d = Upscale2D()
        self.conv0 = WeightScaledConv2D(
            filters=nf(self.res-1),
            kernel_size=3,
        )
        self.conv1 = WeightScaledConv2D(
            filters=nf(self.res-1),
            kernel_size=3,
        )
        self.pixelnorm = PixelNorm()
        super().__init__(*args, **kwargs)

    def call(self, inputs: tf.Tensor):
        x = self.upscale2d(inputs)
        x = self.conv0(x)
        x = self.pixelnorm(x)
        x = self.conv1(x)
        x = self.pixelnorm(x)
        return x


class LastGeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, res: int, growing_factor: tf.Tensor):
        self.res = res
        self.growing_factor = growing_factor
        super().__init__()

        self.to_rgb_1 = ToRGB()
        self.upscale2d = Upscale2D()

        self.block = GeneratorBlock(self.res)
        self.to_rgb_2 = ToRGB()

    def call(self, inputs: tf.Tensor):
        previous_image = self.to_rgb_1(inputs)
        previous_image_upscaled = self.upscale2d(previous_image)

        x = self.block(inputs)
        image = self.to_rgb_2(x)

        # Next line is equivalent to: mix = g * upscaled + (1-g) * x
        mix = previous_image_upscaled + self.growing_factor * \
            (image - previous_image_upscaled)
        return mix


class Generator(tf.keras.models.Sequential):
    def __init__(self, resolution: int, growing_factor: tf.Tensor, latent_dims=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        self.add(tf.keras.layers.InputLayer([latent_dims], dtype=tf.float32))

        self.add(FirstGeneratorBlock())

        for res in range(3,  resolution_log2):
            print(f"Adding a generator block for {2**res}x{2**res}")
            self.add(GeneratorBlock(res, name=f"gen_{2**res}x{2**res}"))

        # last layer
        self.add(
            LastGeneratorBlock(
                resolution_log2,
                growing_factor=growing_factor
            )
        )


def nf(stage,
       # Overall multiplier for the number of feature maps.
       fmap_base=8192,
       # log2 feature map reduction when doubling the resolution.
       fmap_decay=1.0,
       fmap_max=512,          # Maximum number of feature maps in any layer.
       ):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


# Building blocks.
resolution = 64
latents_dims = 128
print(nf(2))
# img = tf.random_normal([3, 3, 64, 64])
x = tf.random_normal([10, 128])
# f1 = PixelNorm()

# TODO: find the right way of integrating the growing factor.
growing_factor = 0.5
growing_factor = tf.placeholder(
    tf.float32,
    [1],
    "growing_factor",
)

g1 = Generator(64, growing_factor)

for layer in g1.layers:
    print(layer.input_shape, layer.output_shape)

g2 = Generator(128, growing_factor)

y1 = g1(x)
y2 = g2(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # y_old = sess.run(y1, feed_dict={growing_factor: [1]})
    # y_new = sess.run(y2, feed_dict={growing_factor: [0.]})

# TODO: Idea: use the intuitive keras methods to transfer the weights.
for i, (layer_old, layer_new) in enumerate(zip(g1.layers, g2.layers)):
    print(i, layer_old.name, layer_new.name)
    layer_new.set_weights(layer_old.get_weights())

# TODO: create the discriminator model, the losses, and the training loop.
