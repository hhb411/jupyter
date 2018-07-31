import tensorflow as tf
import numpy as np


# 交叉熵
def cross_entropy(calculated, real):
    return -tf.reduce_mean(calculated * tf.log(tf.clip_by_value(real, 1e-10, 1.0)) + (1-real) * tf.log(tf.clip_by_value(1-real, 1e-10, 1.0)))

def variable_summary(var, name):
    with tf.name_scope('summary'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(var - mean))
        tf.summary.scalar('stddev/' + name, stddev)

class NN:
    def train(self, X, Y, batch_size=8, steps=5000):
        # 输入
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
            y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
        # 前向传播
        with tf.name_scope('hidden_1'):
            w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
            a = tf.matmul(x, w1)
            variable_summary(w1, 'w1')
        with tf.name_scope('output'):
            w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
            y = tf.matmul(a, w2)
            # 激活函数
            y = tf.sigmoid(y)
            tf.summary.scalar('activation', tf.reduce_mean(y))
        # 损失函数
        with tf.name_scope('loss'):
            loss = cross_entropy(y_, y)
            tf.summary.scalar('loss', tf.reduce_mean(loss))
        # 反向传播
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        merged_summary = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('c:/tensorboard/tf/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('c:/tensorboard/tf/test', tf.get_default_graph())

        # 样本总数
        dataset_size = X.shape[0]
        # 训练
        with tf.Session() as sess:
            # 初始化变量
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("训练前的参数：")
            print(sess.run(w1))
            print(sess.run(w2))

            for i in range(steps):
                # 选取batch_size个样本进行训练
                start = (i * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                train_summary, _ = sess.run([merged_summary, train_step],
                         feed_dict={x: X[start:end], y_: Y[start:end]})
                # 每隔一段时间计算所有数据上的交叉熵
                if i % 100 == 0:
                    # 记录运行时信息
                    train_writer.add_summary(train_summary, i)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    test_summary, test_loss = sess.run([merged_summary, loss], feed_dict={x: X, y_: Y},
                                    options=run_options, run_metadata=run_metadata)

                    test_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    test_writer.add_summary(test_summary, i)
                    print("training step %d, loss on all data is %g" %
                          (i, test_loss))

            print("训练后的参数：")
            print(sess.run(w1))
            print(sess.run(w2))
        train_writer.close()

    def fit(self):
        pass;

# 造训练数据
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [ [int(x1+x2 <1)] for (x1, x2) in X]
# 训练
model = NN()
model.train(X, Y, 8, 5001)
model.fit()