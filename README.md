# (WIP) Keras-Progressive-Growing-GANS
A clean TF-Keras re-implementation of the ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://github.com/tkarras/progressive_growing_of_gans) project.

Most of the components of their `"networks.py"` code use some soon-to-be deprecated TensorFlow API's, like `get_variable()` or `tf.variable_scope()`. 
I made this repo in order to test out the new Tensorflow 2.0 features by re-writing their code such that it is more intuitive and Object-Oriented, using the Keras API with Model Subclassing.

I still have some things to figure out, mostly due to the dynamic growing of the network causing problems in TensorFlow.
