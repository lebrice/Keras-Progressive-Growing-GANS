import tensorflow as tf

from networks_keras import num_filters, Upscale2DConv2D, Upscale2D, ToRGB, PixelNorm, WeightScaledConv2D, WeightScaledDense
from utils import num_filters, resolution_of_stage, stage_of_resolution, assert_valid_resolution


class FirstGeneratorBlock(tf.keras.models.Sequential):
    def __init__(self, *args, **kwargs):
        filters = num_filters(0)
        res = resolution_of_stage(0)
        prefix = f"gen_{res}x{res}"
        super().__init__(layers=[
            PixelNorm(name=f"{prefix}_pixelnorm_1"),
            WeightScaledDense(units=filters * 4 * 4, name=f"{prefix}_dense"),
            tf.keras.layers.Reshape([filters, 4, 4], name=f"{prefix}_reshape"),
            PixelNorm(name=f"{prefix}_pixelnorm_2"),
            WeightScaledConv2D(filters=filters, kernel_size=3,
                               name=f"{prefix}_conv2d"),
            PixelNorm(name=f"{prefix}_output"),
        ], *args, **kwargs)


class GeneratorBlock(tf.keras.models.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape: tf.TensorShape):
        res = input_shape[-1].value
        assert_valid_resolution(res)
        stage = stage_of_resolution(res)
        prefix = f"{2**res}x{2**res}"
        filters = num_filters(stage)
        self.add(
            Upscale2DConv2D(
                filters=filters,
                kernel_size=3,
                name=f"{prefix}_upscale"
            )
        )
        self.add(
            PixelNorm(name=f"{prefix}_pixelnorm")
        )
        self.add(
            WeightScaledConv2D(
                filters=filters,
                kernel_size=3,
                name=f"{prefix}_conv2d"
            )
        )
        self.add(
            PixelNorm(name=f"{prefix}_output")
        )

class Generator(tf.keras.models.Sequential):
    def __init__(self,  latent_dims=512, normalize_latents=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_latents = normalize_latents
        self.add(tf.keras.layers.InputLayer([latent_dims]))
        self.add(FirstGeneratorBlock())
        self.stage: int = 0
        self.resolution: int = resolution_of_stage(self.stage)
        self.images = []
        self._mixing_factor = 0.0

    @property
    def mixing_factor(self) -> tf.Tensor:
        return tf.clip_by_value(self._mixing_factor, 0.0, 1.0)
    
    @mixing_factor.setter
    def mixing_factor(self, value) -> None:
        self._mixing_factor = value

    def call(self, inputs: tf.Tensor, training=False):
        print("call was called inside Generator.", self.stage)
        x = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
        for stage, layer in enumerate(self.layers):
            x = layer(x)
            
            res = resolution_of_stage(stage)
            name = f"gen_image_out_{res}x{res}"
            # Create the output image for each stage, if needed.
            if not hasattr(self, name):
                to_rgb_layer = ToRGB()
                setattr(self, name, to_rgb_layer)
                self.images.append(to_rgb_layer(x))

        new = self.images[-1]
        if (self.stage == 0):
            # we haven't started growing the network yet
            return new

        # Mix the outputs of the last two stages.
        old = self.images[-2]
        old_upscaled = Upscale2D()(old)
        mix = (
            (1 - self.mixing_factor) * old_upscaled +
            self.mixing_factor * new
        )
        return mix

    
    def grow(self) -> None:
        self.stage += 1
        self.resolution *= 2
        print(f"Growing the Generator; output resolution is now {self.resolution}x{self.resolution}")
        print("built:", self.built)
        self.add(GeneratorBlock())
        print("built:", self.built)




# noise = tf.random_normal([1, 512])
# gen_1 = Generator()
# out = gen_1(noise)
# gen_1.summary()
# print(out.shape)
# for _ in range(8):
#     gen_1.grow()
#     print(gen_1.output_shape)
# gen_1.summary()
