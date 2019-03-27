import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np

class Wide_Deep():
	'''
	wide_deep模型TensorFlow Python实现
	'''
	def __init__(self, wide_size, field_size, 
						sparse_size, numeric_size,	
						embedding_size=20, 
						deep_size=[1024, 512, 256],
						epochs=100,
						batch_size=50,
						learning_rate=0.001,
						l2_reg=0.0	
						):

		self.wide_size = wide_size
		self.field_size = field_size
		self.sparse_size = sparse_size
		self.numeric_size = numeric_size
		self.embedding_size = embedding_size
		self.deep_size = deep_size

		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.l2_reg = l2_reg

		self.build_graph()

	def build_graph(self):
		'''初始化网络结构'''

		self.x_wide = tf.placeholder(tf.float32, shape=[None, self.wide_size], name='x_wide')
		self.x_deep_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name='x_deep_index')
		self.x_deep_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name='x_deep_value')
		self.x_deep_numeric = tf.placeholder(tf.float32, shape=[None, self.numeric_size], name='x_deep_numeric')
		self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

		# wide component
		# 初始化wide 的权重
		w_wide = tf.Variable(tf.random_normal(shape=[self.wide_size, 1]), name='w_wide')
		logit_wide = tf.matmul(self.x_wide, w_wide)

		## deep component
		w_embedding = tf.Variable(tf.random_normal(shape=[self.sparse_size, self.embedding_size],
													mean=0, stddev=0.01), name='w_embedding')
		lookup = tf.nn.embedding_lookup(w_embedding, self.x_deep_index)
		x_deep_value = tf.reshape(self.x_deep_value, [-1, self.field_size, 1])
		dense_embedding = tf.multiply(lookup, x_deep_value) # None*(embedding_size*field_size)
		dense_embedding = tf.reshape(dense_embedding, [-1, self.embedding_size*self.field_size])
		# 将离散特征的嵌入和连续特征连接起来作为网络的输入
		input_deep = tf.concat([dense_embedding, self.x_deep_numeric], axis=1)

		self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
		bn_params = {
		'is_training': self.is_training,
		'decay': 0.995,
		'updates_collections': None
		}
		for i in range(len(self.deep_size)):
			hidden_bn = tf.contrib.layers.fully_connected(inputs=input_deep,
														num_outputs=self.deep_size[i],
														normalizer_fn=tf.contrib.layers.batch_norm,
														normalizer_params=bn_params,
														weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
			input_deep = hidden_bn

		# deep 侧输出层权重和bias
		w_deep_output = tf.Variable(tf.random_normal(shape=[self.deep_size[-1], 1]), name='w_deep_output', dtype=tf.float32)
		bias = tf.Variable(tf.zeros([1], dtype=tf.float32))
		logit_deep = tf.matmul(input_deep, w_deep_output)

		logit = logit_wide + logit_deep + bias
		logit = tf.reshape(logit,[-1,1])
		self.y_pred = tf.sigmoid(logit)

		# loss
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=self.y)
		self.loss = tf.reduce_mean(entropy) + self.l2_reg*(tf.nn.l2_loss(w_embedding) + tf.nn.l2_loss(w_deep_output))

		# optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
												beta1=0.9, beta2=0.999, epsilon=1e-8)
		self.train_op = self.optimizer.minimize(self.loss)

		# 建立一个session
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def train_model(self, x_wide, x_deep_numeric, x_deep_index, x_deep_value, labels):
		'''模型训练'''
		for epoch in range(self.epochs):
			for x_wide_batch, x_deep_numeric_batch, x_deep_index_batch, x_deep_value_batch, labels_batch in \
			self.random_shuffle_batch(x_wide, x_deep_numeric, x_deep_index, x_deep_value, labels):
				self.sess.run(self.train_op, feed_dict={self.x_wide : x_wide_batch,
														self.x_deep_numeric : x_deep_numeric_batch,
														self.x_deep_index : x_deep_index_batch,
														self.x_deep_value : x_deep_value_batch,
														self.y : labels_batch})
			loss, auc = self.evaluate(x_wide, x_deep_numeric, x_deep_index, x_deep_value, labels)
			print(epoch, 'loss:',loss,'auc',auc)


	def evaluate(self, x_wide, x_deep_numeric, x_deep_index, x_deep_value, labels):
		'''模型评估'''
		y_pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={self.x_wide : x_wide,
														self.x_deep_numeric : x_deep_numeric,
														self.x_deep_index : x_deep_index,
														self.x_deep_value : x_deep_value,
														self.y : labels})
		auc = roc_auc_score(labels, y_pred)
		return loss, auc


	def predict(self, x_wide, x_deep_numeric, x_deep_index, x_deep_value):
		'''预测'''
		y_pred = self.sess.run(self.y_pred, feed_dict={self.x_wide : x_wide,
														self.x_deep_numeric : x_deep_numeric,
														self.x_deep_index : x_deep_index,
														self.x_deep_value : x_deep_value,
														})
		return y_pred


	def random_shuffle_batch(self, x_wide, x_deep_numeric, x_deep_index, x_deep_value, labels):
		'''随机批量采样'''
		rnd_index = np.random.permutation(len(labels))
		n_batchs = len(labels) // self.batch_size
#		print(rnd_index, n_batchs)
		for idx in np.array_split(rnd_index, n_batchs):
			x_wide_batch, x_deep_numeric_batch, x_deep_index_batch, x_deep_value_batch, labels_batch = \
			x_wide[idx], x_deep_numeric[idx], x_deep_index[idx], x_deep_value[idx], labels[idx]
			yield x_wide_batch, x_deep_numeric_batch, x_deep_index_batch, x_deep_value_batch, labels_batch
