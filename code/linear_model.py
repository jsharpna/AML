### The following code and text is extracted from https://www.tensorflow.org/guide/graphs 
### Any relevant copywrite applies

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## Silly data
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

"""
This layer implements the operation: outputs = activation(inputs * kernel + bias) Where activation is the activation function passed as the activation argument (if not None), kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True).

The layer is an operation with variables (to be seen)
"""
linear_model = tf.layers.Dense(units=1)


"""
Apply the linear_model operation to tensor x to produce tensor y_pred
"""
y_pred = linear_model(x)


"""
Losses are output scalar tensors and can accept weights, etc.
"""
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

"""
Can either use a scalar learning rate or tensor (I want to change mine so I use a tensor)
"""
learning_rate = tf.placeholder(tf.float32, shape=[])


"""
train module contains optimizers such as AdaGrad and Adam
- contains methods: apply_gradients, compute_gradients, variables, minimize
- minimize returns an operation that minimizes the input (loss) tensor 
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

"""
You can initialize variables by hand or just use random starts like below
"""
init = tf.global_variables_initializer()


"""
TensorFlow uses the tf.Session class to represent a connection between the client program---typically a Python program, although a similar interface is available in other languages---and the C++ runtime. A tf.Session object provides access to devices in the local machine, and remote devices using the distributed TensorFlow runtime. It also caches information about your tf.Graph so that you can efficiently run the same computation multiple times.

- the session should be closed to free up the device, or even better, use the with clause.
- the run method will execute the subgraph in order to evaluate the tensor or run the operation.
"""

with tf.Session() as sess:
    sess.run(init)

    for i in range(100):
        lr = .5*(i+1)**-1.
        _, loss_value = sess.run((train, loss), feed_dict={learning_rate : lr})
        print(loss_value)

    print(sess.run(y_pred))
    print(sess.run(linear_model.weights))


