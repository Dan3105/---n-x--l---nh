import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras

IMAGE_SIZE = (224, 224)

def get_encoder():
  vgg19 = keras.applications.VGG19(
      include_top=False,
      weights="imagenet",
      input_shape=(*IMAGE_SIZE, 3)
  )
  vgg19.trainable = False
  encoder = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

  inputs = layers.Input([*IMAGE_SIZE, 3])
  encoder_output = encoder(inputs)
  return keras.Model(inputs, encoder_output, name="encoder_model")

def get_decoder():
  config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
  decoder = keras.Sequential([
      layers.InputLayer((None, None, 512)),
      layers.Conv2D(512, **config),
      layers.UpSampling2D(),
      layers.Conv2D(filters=256, **config),
      layers.BatchNormalization(),
      layers.Conv2D(filters=256, **config),
      layers.BatchNormalization(),
      layers.Conv2D(filters=256, **config),
      layers.UpSampling2D(),
      layers.Conv2D(filters=128, **config),
      layers.BatchNormalization(),
      layers.Conv2D(filters=128, **config),
      layers.UpSampling2D(),
      layers.Conv2D(filters=64, **config),
      layers.Conv2D(
          filters=3,
          kernel_size=3,
          strides=1,
          padding="same",
          activation="sigmoid"
      )
  ])
  return decoder

def get_loss_net():
  vgg19 = keras.applications.VGG19(
      include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
  )
  vgg19.trainable = False
  layer_names=["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
  outputs = [vgg19.get_layer(name).output for name in layer_names]

  mini_vgg19 = tf.keras.Model(vgg19.input, outputs)
  inputs= layers.Input([*IMAGE_SIZE, 3])
  mini_vgg19_out = mini_vgg19(inputs)
  return keras.Model(inputs, mini_vgg19_out, name="loss_net")

def load_saved_model(dir):
  return tf.saved_model.load(dir)

def get_mean_std(x, epsilon=1e-5):
  """
  x: (n, w, h, c)
  """
  axes = [1,2]
  mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
  std = tf.sqrt(variance + epsilon)
  return mean, std

def ada_in(style, content):
  """
  style: (n, w, h, c) -> encoder output
  content: (n, w, h, c) -> //

  return -> T=(n, w, h, c)
  """

  content_mean, content_std = get_mean_std(content)
  style_mean, style_std = get_mean_std(style)
  t = style_std * (content - content_mean) / content_std + style_mean
  print(t.get_shape())
  return t

class NeuralStyleTransfer(tf.keras.Model):
  def __init__(self, encoder, decoder, loss_net, style_weight, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.loss_net = loss_net
    self.style_weight = style_weight

  def compile(self, optimizer, loss_fn):
    super().compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
    self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

  def train_step(self, input):
    style, content = input
    loss_content = 0.0
    loss_style = 0.0

    with tf.GradientTape() as tape:
      style_encode = self.encoder(style)
      content_encode = self.encoder(content)

      t = ada_in(style=style_encode, content=content_encode)
      t_output = self.decoder(t)

      #Compute loss
      t_features = self.loss_net(t_output) #extract from specific type for calculating
      style_features = self.loss_net(style)
      loss_content = self.loss_fn(t_features[-1], t)
      for style_feature, t_feature in zip(style_features, t_features):
        mean_style, std_style = get_mean_std(style_feature)
        mean_t, std_t = get_mean_std(t_feature)

        loss_style += self.loss_fn(mean_style, mean_t) + self.loss_fn(std_style, std_t)
      loss_style = self.style_weight * loss_style
      total_loss = loss_content + loss_style

    trainable_vars = self.decoder.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.style_loss_tracker.update_state(loss_style)
    self.content_loss_tracker.update_state(loss_content)
    self.total_loss_tracker.update_state(total_loss)

    return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

  def test_step(self, input):
    style, content = input

    loss_content = 0.0
    loss_style = 0.0

    style_encode = self.encoder(style)
    content_encode = self.encoder(content)

    t = ada_in(style_encode, content_encode)

    result = self.decoder(t)

    #Compute loss
    t_features = self.loss_net(result) #extract from specific type for calculating
    style_features = self.loss_net(style)
    loss_content = self.loss_fn(t_features[-1], t)
    for style_feature, t_feature in zip(style_features, t_features):
      mean_style, std_style = get_mean_std(style_feature)
      mean_t, std_t = get_mean_std(t_feature)

      loss_style += self.loss_fn(mean_style, mean_t) + self.loss_fn(std_style, std_t)
    loss_style = self.style_weight * loss_style
    total_loss = loss_content + loss_style

    # Update the trackers.
    self.style_loss_tracker.update_state(loss_style)
    self.content_loss_tracker.update_state(loss_content)
    self.total_loss_tracker.update_state(total_loss)

    return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

  #accept only image size 224,224, 3
  def predict(self, style_image, content_image):
    style_image = np.expand_dims(style_image, axis=0)
    content_image = np.expand_dims(content_image, axis=0)


    style_encode = self.encoder(style_image)
    content_encode = self.encoder(content_image)

    t = ada_in(style_encode, content_encode)

    result = self.decoder(t)

    return result[0]

  @property
  def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]
  
def load_neural_model(dir):
  """
  dir: pretrained decoder model
  """
  encoder = get_encoder()
  decoder = load_saved_model(dir)
  loss_net = get_loss_net()

  my_model = NeuralStyleTransfer(encoder, decoder, loss_net, style_weight=2.0)

  return my_model