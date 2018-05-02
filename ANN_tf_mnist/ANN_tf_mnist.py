import tensorflow as tf 


############################ Tunable Parameters #################################

learning_rate = 0.1
n_epochs = 500
batch_size = 100
display_step = 20

n_input=784									  	# number of input neurons (ignoring the bias neuron) - 784  for MNIST
n_output=10										# number of output neurons - 10  for digit recognition
n_neurons=[n_input,400,200,n_output]			# the list of neurons in successive hidden layers


#################################################################################


def train_model():
	'''
	This function is quite generic- it could be used for most situations by tuning the above 
	parameters
	'''

	# DECLARING the fixed (un-trainable) vaiables - the "Placeholders" of TF :
	X=tf.placeholder(tf.float32,shape=(None,n_input))
	Y=tf.placeholder(tf.float32,shape=(None,n_output))	

	# DECLARING the trainable variables -  the "Variables" of TF :
	
	layer_tuples=zip(n_neurons[:-1],n_neurons[1:])

	weights=[tf.Variable(tf.random_normal(layer_tuple)) for layer_tuple in layer_tuples]
	biases=[tf.Variable(tf.random_normal([val])) for val in n_neurons[1:]]

	# CREATING THE MODEL:
	a=X
	for w,b in zip(weights,biases):
		a=tf.sigmoid(tf.matmul(a,w)+b)

	y=a
	
	# the loss function to reduce. NOTE that "reduce_mean or reduce_sum" have nothing to do with learning process: 
	# use the first loss function ---- preferably
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=Y))
	#cross_entropy=tf.reduce_mean(tf.losses.mean_squared_error(Y,tf.nn.softmax(y)))

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

	# the Session object initialized
	sess = tf.Session()


	# initializing the global variables.NOTE that they still have to be given the right value at each step
	tf.global_variables_initializer().run(session=sess)

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


	# finding max number of batches:
	n_batches = int(mnist.train.num_examples/float(batch_size))

	# data for plotting graphs:


	#############################  The real training stage begins here ##############################################

	for j in range(n_epochs):
		for i in range(n_batches):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

		if j%display_step==0:
			# finding E_in and E_out at the end of each epoch
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			E_in=1-sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels})
			E_out=1-sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
			print "epoch:",j,"; E_in:",
			print("{0:.4f}".format(E_in)),"; E_out:",
			print("{0:.4f}".format(E_out)) 



	####################################################################################################################

train_model()
