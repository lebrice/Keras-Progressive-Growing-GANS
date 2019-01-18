from typing import NamedTuple, Type, List, Tuple, Iterable
from functools import wraps

import numpy as np
import tensorflow as tf
import os
from utils import log2, HeInitializer

tf.enable_eager_execution()
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
        super().__init__(units=units, activation=activation,
                         use_bias=use_bias, *args, **kwargs)


class WeightScaledConv2D(WeightScaled(tf.keras.layers.Conv2D)):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 activation=tf.nn.leaky_relu,
                 strides=(1, 1),
                 data_format="channels_first",
                 padding="SAME",
                 use_bias=True,
                 *args, **kwargs):
        super().__init__(
            filters=filters, kernel_size=kernel_size, activation=activation,
            strides=strides, data_format=data_format, padding=padding, *args, **kwargs)


class Conv2DDownscale2D(WeightScaledConv2D):
    def __init__(self, filters: int, kernel_size: int, strides=2, *args, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size,
                         strides=strides, *args, **kwargs)


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


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

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([*input_shape[:-2], *(d.value * 2 for d in input_shape[-2:])])


class Upscale2DConv2D(WeightScaled(tf.keras.layers.Conv2DTranspose)):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="SAME",
        data_format="channels_first",
        *args,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding=padding,
            data_format=data_format,
            *args,
            **kwargs,
        )


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
        self.to_rgb = ToRGB()
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor):
        if self.normalize_latents:
            x = self.pixelnorm(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.pixelnorm(x)
        x = self.conv(x)
        x = self.pixelnorm(x)
        # TODO: Does the first block need an image output?
        self.image_out = self.to_rgb(x)
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0].value, 512, 4, 4])


def first_generator_block() -> List[tf.keras.layers.Layer]:
    return [
        PixelNorm(name="gen_4x4_pixelnorm_1"),
        WeightScaledDense(units=512 * 4 * 4, name="gen_4x4_dense"),
        tf.keras.layers.Reshape([512, 4, 4], name="gen_4x4_reshape"),
        PixelNorm(name="gen_4x4_pixelnorm_2"),
        WeightScaledConv2D(filters=512, kernel_size=3, name="gen_4x4_conv2d"),
        PixelNorm(name="gen_4x4_output"),
    ]


class GeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, res_log2: int, *args, **kwargs):
        self.res_log2 = res_log2
        self.filters = num_filters(self.res_log2-1)
        if "name" not in kwargs:
            kwargs["name"] = f"generator_block_{2**res_log2}x{2**res_log2}"
        self.upscale_conv = Upscale2DConv2D(
            filters=self.filters,
            kernel_size=3,
        )
        self.conv1 = WeightScaledConv2D(
            filters=self.filters,
            kernel_size=3,
        )
        self.to_rgb = ToRGB()
        super().__init__(*args, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # x = self.upscale2d(inputs)
        # x = self.conv0(x)
        x = self.upscale_conv(inputs)
        x = PixelNorm()(x)
        x = self.conv1(x)
        x = PixelNorm()(x)
        self.image_out = self.to_rgb(x)
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0], self.filters, 2**self.res_log2, 2**self.res_log2])


def generator_block(res: int) -> List[tf.keras.layers.Layer]:
    prefix = f"{2**res}x{2**res}"
    filters = num_filters(res-1)
    return [
        Upscale2DConv2D(filters=filters, kernel_size=3,
                        name=f"{prefix}_upscale"),
        PixelNorm(name=f"{prefix}_pixelnorm"),
        WeightScaledConv2D(filters=filters, kernel_size=3,
                           name=f"{prefix}_conv2d"),
        PixelNorm(name=f"{prefix}_output"),
    ]


def make_generator(output_resolution: int, latent_dims=128, growing_factor=0.0) -> tf.keras.Model:
    res_log2 = log2(output_resolution)

    blocks = []
    latents = tf.keras.Input([latent_dims])
    x = latents
    blocks = [FirstGeneratorBlock()] + [GeneratorBlock(res)
                                        for res in range(3, res_log2+1)]
    # print(blocks)

    for block in blocks:
        x = block(x)
    images = [block.image_out for block in blocks]

    old = images[-2]
    new = images[-1]

    old_upscaled = Upscale2D()(old)

    mix = new + growing_factor * (old_upscaled - new)
    # output = tf.identity(mix, "image_out")
    return mix


class Generator(tf.keras.Sequential):
    def __init__(self, image_resolution: int=4, latent_dims=128, growing_factor: tf.Tensor = 0.0):
        super().__init__()
        self.res_log2 = log2(image_resolution)
        self.growing_factor = growing_factor

        self.add(tf.keras.layers.InputLayer([latent_dims]))
        for layer in first_generator_block():
            self.add(layer)
        for res in range(3, self.res_log2 + 1):
            for layer in generator_block(res):
                self.add(layer)
        self.upscale = Upscale2D()
        self.images = []

    @property
    def stage(self) -> int:
        return self.res_log2 - 1

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        print("call is called in generator")
        for layer in self.layers:
            x = layer(x)
            if "_output" in layer.name and not hasattr(self, layer.name + "_image"):
                image_layer = ToRGB()
                setattr(self, layer.name + "_image", image_layer)
                self.images.append(image_layer(x))
        print(self.images)
        new = self.images[-1]
        if (self.stage == 1):
            return new
            # we haven't started growing the network yet
        # print(self.images)
        old = self.images[-2]

        old_upscaled = self.upscale(old)
        mix = old_upscaled * (1 - self.growing_factor) + new * self.growing_factor
        return mix


    def grow(self) -> None:
        """Grows the generator, adding in another stage and doubling the output resolution."""
        self.growing_factor = 0
        self.res_log2 += 1
        for layer in generator_block(self.res_log2):
            self.add(layer)

def num_filters(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    """Gives the number of feature maps that is reasonable for a given stage.
    Arguments:
        fmap_base: Overall multiplier for the number of feature maps.        
        fmap_decay: log2 feature map reduction when doubling the resolution.        
        fmap_max: Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def test_discriminator(resolution: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        [3, resolution, resolution], dtype=tf.float32))
    model.add(tf.keras.layers.Conv2D(64, 3, data_format="channels_first"))
    model.add(tf.keras.layers.Conv2D(
        64, 3, data_format="channels_first", strides=2))
    model.add(tf.keras.layers.Conv2D(
        64, 3, data_format="channels_first", strides=2))
    model.add(tf.keras.layers.Conv2D(
        64, 3, data_format="channels_first", strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model



# x = tf.random_normal([1, 128])
# def show_image(generator):
#     y = generator(x)
#     from utils import NCHW_to_NHWC, unnormalize_images
#     y = NCHW_to_NHWC(y)
#     y = unnormalize_images(y)
#     print(y)
#     import matplotlib.pyplot as plt
#     plt.imshow(y[0].numpy())
#     plt.show(1)

# gen = Generator()
# show_image(gen)
# gen.grow()
# show_image(gen)
# for _ in range(10):
#     gen.growing_factor += 0.05
#     show_image(gen)