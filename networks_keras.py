from typing import NamedTuple, Type, List, Tuple, Iterable, Dict
from functools import wraps

import numpy as np
import tensorflow as tf
import os
from utils import log2, HeInitializer, num_filters

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


class Downscale2D(tf.keras.layers.AveragePooling2D):
    def __init__(self, *args, **kwargs):
        super().__init__(data_format="channels_first", *args, **kwargs)


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
            activation=tf.keras.activations.linear,
            *args,
            **kwargs,
        )


class FromRGB(WeightScaledConv2D):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(filters=filters, kernel_size=1, *args, **kwargs)


class MinibatchStdDevLayer(tf.keras.layers.Layer):
    def __init__(self, group_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        # [NCHW]  Input shape.
        s = x.shape
        # [GMCHW] Split minibatch into M groups of size G.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
        # [GMCHW] Cast to FP32.
        y = tf.cast(y, tf.float32)
        # [GMCHW] Subtract mean over group.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        # [MCHW]  Calc variance over group.
        y = tf.reduce_mean(tf.square(y), axis=0)
        # [MCHW]  Calc stddev over group.
        y = tf.sqrt(y + 1e-8)
        # [M111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        # [M111]  Cast back to original data type.
        y = tf.cast(y, x.dtype)
        # [N1HW]  Replicate over group and pixels.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        # [NCHW]  Append as new fmap.
        return tf.concat([x, y], axis=1)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        out = input_shape.as_list()
        out[-3] += 1
        return tf.TensorShape(out)


# class FirstGeneratorBlock(tf.keras.layers.Layer):
#     def __init__(self, normalize_latents: bool = True, pixelnorm_epsilon=1e-8, *args, **kwargs):
#         self.normalize_latents = normalize_latents
#         self.pixelnorm = PixelNorm(pixelnorm_epsilon)
#         self.dense = WeightScaledDense(
#             units=512 * 4 * 4,
#         )
#         self.reshape = tf.keras.layers.Reshape([512, 4, 4])
#         self.conv = WeightScaledConv2D(
#             filters=512,
#             kernel_size=3,
#         )
#         self.to_rgb = ToRGB()
#         super().__init__(*args, **kwargs)

#     def call(self, x: tf.Tensor):
#         if self.normalize_latents:
#             x = self.pixelnorm(x)
#         x = self.dense(x)
#         x = self.reshape(x)
#         x = self.pixelnorm(x)
#         x = self.conv(x)
#         x = self.pixelnorm(x)
#         # TODO: Does the first block need an image output?
#         self.image_out = self.to_rgb(x)
#         return x

#     def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
#         return tf.TensorShape([input_shape[0].value, 512, 4, 4])


# class GeneratorBlock(tf.keras.layers.Layer):
#     def __init__(self, res_log2: int, *args, **kwargs):
#         self.res_log2 = res_log2
#         self.filters = num_filters(self.res_log2-1)
#         if "name" not in kwargs:
#             kwargs["name"] = f"generator_block_{2**res_log2}x{2**res_log2}"
#         self.upscale_conv = Upscale2DConv2D(
#             filters=self.filters,
#             kernel_size=3,
#         )
#         self.conv1 = WeightScaledConv2D(
#             filters=self.filters,
#             kernel_size=3,
#         )
#         self.to_rgb = ToRGB()
#         super().__init__(*args, **kwargs)

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         # x = self.upscale2d(inputs)
#         # x = self.conv0(x)
#         x = self.upscale_conv(inputs)
#         x = PixelNorm()(x)
#         x = self.conv1(x)
#         x = PixelNorm()(x)
#         self.image_out = self.to_rgb(x)
#         return x

#     def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
#         return tf.TensorShape([input_shape[0], self.filters, 2**self.res_log2, 2**self.res_log2])


def first_generator_block() -> List[tf.keras.layers.Layer]:
    return [
        PixelNorm(name="gen_4x4_pixelnorm_1"),
        WeightScaledDense(units=512 * 4 * 4, name="gen_4x4_dense"),
        tf.keras.layers.Reshape([512, 4, 4], name="gen_4x4_reshape"),
        PixelNorm(name="gen_4x4_pixelnorm_2"),
        WeightScaledConv2D(filters=512, kernel_size=3, name="gen_4x4_conv2d"),
        PixelNorm(name="gen_4x4_output"),
    ]



def generator_block(res: int) -> List[tf.keras.layers.Layer]:
    prefix= f"{2**res}x{2**res}"
    filters= num_filters(res-1)
    return [
        Upscale2DConv2D(filters=filters, kernel_size=3,
                        name=f"{prefix}_upscale"),
        PixelNorm(name=f"{prefix}_pixelnorm"),
        WeightScaledConv2D(filters=filters, kernel_size=3,
                           name=f"{prefix}_conv2d"),
        PixelNorm(name=f"{prefix}_output"),
    ]

class Generator(tf.keras.models.Sequential):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

# class Generator(tf.keras.Sequential):
#     def __init__(self, image_resolution: int = 4, latent_dims=128, growing_factor: tf.Tensor = 0.0):
#         super().__init__()
#         self.res_log2= log2(image_resolution)
#         self.growing_factor= growing_factor

#         self.add(tf.keras.layers.InputLayer([latent_dims]))
#         for layer in first_generator_block():
#             self.add(layer)
#         for res in range(3, self.res_log2 + 1):
#             for layer in generator_block(res):
#                 self.add(layer)
#         self.upscale= Upscale2D()
#         self.images= []
#         # self.image_layers = []

#     @property
#     def stage(self) -> int:
#         return self.res_log2 - 1

#     def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
#         print("call is called in generator")
#         for layer in self.layers:
#             x= layer(x)
#             if "_output" in layer.name and not hasattr(self, layer.name + "_image"):
#                 image_layer= ToRGB()
#                 # self.image_layers.append(image_layer)
#                 setattr(self, layer.name + "_image", image_layer)
#                 self.images.append(image_layer(x))

#         new= self.images[-1]
#         if (self.stage == 1):
#             # we haven't started growing the network yet
#             return new

#         old= self.images[-2]
#         old_upscaled= self.upscale(old)
#         mix= (
#             (1 - self.growing_factor) * old_upscaled +
#             self.growing_factor * new
#         )
#         return mix

#     def grow(self) -> None:
#         """Grows the generator, adding in another stage and doubling the output resolution."""
#         self.growing_factor = 0
#         self.res_log2 += 1
#         prefix = f"{2**self.res_log2}x{2**self.res_log2}"
#         filters = num_filters(self.res_log2-1)

#         self.add(Upscale2DConv2D(filters=filters, kernel_size=3,name=f"{prefix}_upscale"))
#         self.add(PixelNorm(name=f"{prefix}_pixelnorm"))
#         self.add(WeightScaledConv2D(filters=filters, kernel_size=3,name=f"{prefix}_conv2d"))
#         self.add(PixelNorm(name=f"{prefix}_output"))

#         # for layer in generator_block(self.res_log2):
#             # self.add(layer)


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





# trained_weights = discriminator.get_weights()
# trained_config = discriminator.get_config()
# print(trained_config)
# discriminator2 = Discriminator.from_config(discriminator.get_config(), custom_objects={
#     "FromRGB": FromRGB
# })

# x2 = tf.random_normal([1, 3, 1024, 1024])
# disc_2 = Discriminator()
# # disc_2.set_weights(disc_1.get_weights)
# y2 = disc_2(x2)
# disc_2.summary()
# print(y2.shape)
