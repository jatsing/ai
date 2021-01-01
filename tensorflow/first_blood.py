import tensorflow as tf
tf.enable_eager_execution()
x = tf.constant([[[1,2,3.0]]])/2.0
# tf.Tensor([[[0.5 1.  1.5]]], shape=(1, 1, 3), dtype=float32)

mnist = tf.keras.datasets.mnist

# (60000, 28, 28)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  # 将输入28 * 28，拍平 -> 784维
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # 784 -> 128
  tf.keras.layers.Dense(128, activation='relu'),
  # 20% rate断开神经元节点之间的链接
  tf.keras.layers.Dropout(0.2),
  # 128 -> 10
  tf.keras.layers.Dense(10, activation='softmax')
])

# 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
model.compile(optimizer='adam', # sgd, adagrad, adadelta, adam
              loss='sparse_categorical_crossentropy', # mse, sparse_categorical_crossentropy
              metrics=['accuracy']) # accuracy, sparse_accuracy, sparse_categorical_accuracy

# 用于执行训练，指定迭代次数
model.fit(x_train, y_train, epochs=5)

# predict和eval区别：eval会有label,loss计算等，predict是单纯的预估 
# verbose=1, #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
model.evaluate(x_test,  y_test, verbose=2)