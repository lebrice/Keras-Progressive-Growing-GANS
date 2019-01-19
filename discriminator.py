import tensorflow as tf
from networks_keras import WeightScaledConv2D, WeightScaledDense, num_filters, Downscale2D, MinibatchStdDevLayer, FromRGB
from utils import log2, stage_of_resolution, resolution_of_stage
from typing import List


class DiscriminatorBlock(tf.keras.models.Sequential):
    def __init__(self, resolution: int, kernel_size=3, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = f"disc_block_{resolution}x{resolution}" 
        super().__init__(*args, **kwargs)
        self.stage = stage_of_resolution(resolution)
        self.add(WeightScaledConv2D(filters=num_filters(self.stage+1), kernel_size=3, name=f"disc_{self.stage}_conv0"))
        self.add(WeightScaledConv2D(filters=num_filters(self.stage), kernel_size=3, name=f"disc_{self.stage}_conv1"))
        self.add(Downscale2D(name=f"disc_{self.stage}_avg_pooling"))


class Discriminator(tf.keras.models.Sequential):
    def __init__(self, resolution: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resolution: int = resolution
        self.layers: List[tf.keras.layers.Layer]
        self._mixing_factor: float = 0.0
        if self.resolution is not None:
            self.build(tf.TensorShape([3,self.resolution, self.resolution]))

    def build(self, input_shape: tf.TensorShape):
        self.clear()

        self.resolution = input_shape[-1].value
        self.stage = stage_of_resolution(self.resolution)

        self.add(tf.keras.layers.InputLayer([3, self.resolution, self.resolution]))
        self.add(FromRGB(num_filters(self.stage), name=f"disc_{self.stage}_from_rgb"))
        
        for i in range(self.stage, 0, -1):
            self.add(DiscriminatorBlock(resolution_of_stage(i)))
            from_rgb_layer_name = f"disc_{i}_from_rgb"
            if not hasattr(self, from_rgb_layer_name):
                from_rgb_layer = FromRGB(num_filters(self.stage), name=from_rgb_layer_name)
                setattr(self, from_rgb_layer_name, from_rgb_layer)

        self.add(MinibatchStdDevLayer())
        self.add(WeightScaledConv2D(filters=num_filters(1), kernel_size=3, name="conv0"))
        self.add(WeightScaledConv2D(filters=num_filters(1), kernel_size=4, padding="valid", name="conv1"))
        self.add(tf.keras.layers.Flatten())
        self.add(WeightScaledDense(units=1, gain=1, activation=tf.keras.activations.linear, name="dense0"))
        super().build(input_shape)

    @property
    def mixing_factor(self) -> tf.Tensor:
        return tf.clip_by_value(self._mixing_factor, 0.0, 1.0)
    
    @mixing_factor.setter
    def mixing_factor(self, value) -> None:
        self._mixing_factor = value

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        if self.stage == 0:
            # we have a 4x4 input resolution. No need to mix anything.
            # the output is just the chaining of the layers.
            return super().call(inputs, training)
        
        downscaled_inputs = Downscale2D()(inputs)
        from_rgb_previous = getattr(self, f"disc_{self.stage}_from_rgb")
        downscaled_features = from_rgb_previous(downscaled_inputs)
        
        from_rgb_current = self.layers[0]
        # also equivalent:
        # from_rgb_current = getattr(self, f"disc_{self.stage}_from_rgb")
        first_block = self.layers[1]

        features = from_rgb_current(inputs)
        first_block_output = first_block(features)
        
        x = (
            (1-self.mixing_factor) * downscaled_features +
            self.mixing_factor * first_block_output
        )
        print(x.shape)

        for layer in self.layers[2:]:
            x = layer(x)

        return x
    
    def clear(self):
        """Remove all layers. TODO: check if this is working."""
        while len(self.layers) != 0:
            self.pop()

    def grow(self) -> "Discriminator":
        """
        Creates a bigger discriminator, copies the current weights over, and returns it.
        """
        new = Discriminator(self.resolution * 2)
        current_layers = {
            l.name: l for l in self.layers 
        }
        for layer in new.layers:
            if layer.name in current_layers:
                print(f"layer '{layer.name}' is common.")
                layer.set_weights(current_layers[layer.name].get_weights())
        return new


# x = tf.random_normal([1, 3, 8, 8])
# disc_1 = Discriminator()
# y = disc_1(x)
# disc_1.summary()

# x_2 = tf.random_normal([1,3,16,16])
# disc_2 = disc_1.grow()
# y_2 = disc_2(x_2)
# disc_2.summary()