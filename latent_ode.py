import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

class RecognitionRNN(tf.keras.Model):
	"""TODO: Description """
    def __init__(self, latent_dim, nhidden):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.i2h = tf.keras.layers.Dense(nhidden, activation="tanh")
        self.h2o = tf.keras.layers.Dense(latent_dim * 2)

    def call(self, inputs, **kwargs):
        h = self.i2h(inputs)
        out = self.h2o(h)
        return out, h

class ODEModel(tf.keras.Model):
	"""TODO: Description """
    def __init__(self, latent_dim, nhidden):
        super(ODEModel, self).__init__()
        self.elu = tf.keras.layers.ELU()
        self.linear1 = tf.keras.layers.Dense(nhidden)
        self.linear2 = tf.keras.layers.Dense(nhidden)
        self.linear3 = tf.keras.layers.Dense(latent_dim)

    def call(self, inputs, **kwargs):
        t, y = inputs
        out = self.linear1(y)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        return out

class BasicLatentODE(LatentODE):
	"""TODO: Description """
	def __init__(self, latent_dim=4, ode_Dh=20, inference_Dh=25, generative_Dh=20, out_dim):
		
		self.latent_dim = latent_dim
		
		# Create basic inference, ode, and generative nets with the latent dimensions, 
		# and number of hidden units given
		
		inference_net = RecognitionRNN(latent_dim, inference_Dh)
		ode_net = ODEModel(latent_dim, ode_Dh)
		generative_net = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(generative_Dh),
				tf.keras.layers.ReLU(),
				tf.keras.layers.Dense(out_dim)
			]
		)
		
		# Initialize the latent ode
		super(BasicLatentODE, self).__init__(inference_net, ode_net, generative_net)
		

class LatentODE(tf.keras.Model):
	"""TODO: Description """
	def __init__(self, inference_net: tf.keras.Model, ode_net: tf.keras.Model,
		generative_net: tf.keras.Model):
		
		super(LatentODE, self).__init__()

	self.inference_net = inference_net
	self.ode_net = ode_net
	self.generative_net = generative_net

	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, z, apply_sigmoid=False):
		logits = self.generative_net(z)
		if apply_sigmoid:
		  probs = tf.sigmoid(logits)
		  return probs

		return logits



optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(
		-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), 
		axis=raxis)

def compute_loss(model, x):
	mean, logvar = model.encode(x)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)

	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
	logpz = log_normal_pdf(z, 0., 0.)
	logqz_x = log_normal_pdf(z, mean, logvar)
	return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)
	return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
	optimizer.apply_gradients(zip(gradients, variables))

