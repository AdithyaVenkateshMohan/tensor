import sys
sys.path.append('C:\python64\Lib\site-packages')
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("/tmp/data/", one_hot= True)

no_nodes_hl1=500
no_nodes_hl2=200
no_nodes_hl3=300

n_classes=10
batch_size = 100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def Neural_Network_model(data) :
    hidden_Layer_1={'weights': tf.Variable(tf.random_normal([784,no_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([no_nodes_hl1]))}

    hidden_Layer_2={'weights':tf.Variable(tf.random_normal([no_nodes_hl1,no_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([no_nodes_hl2]))}

    hidden_Layer_3 = {'weights': tf.Variable(tf.random_normal([no_nodes_hl2, no_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([no_nodes_hl3]))}

    output_Layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    layer1 = tf.add(tf.matmul(data,hidden_Layer_1['weights']),hidden_Layer_1['biases'] )
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1, hidden_Layer_2['weights']), hidden_Layer_2['biases'])
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.add(tf.matmul(layer2, hidden_Layer_3['weights']),hidden_Layer_3['biases'])
    layer3 = tf.nn.relu(layer3)

    output= tf.matmul(layer3, output_Layer['weights']) + output_Layer['biases']

    return output

def train_neural_networks(x):
    predict = Neural_Network_model(x)
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y) )
    optimize = tf.train.AdamOptimizer().minimize(cost)
    Total_epoch=10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(Total_epoch):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x,e_y = mnist.train.next_batch(batch_size)
                _,c=sess.run([optimize,cost], feed_dict={x : e_x , y : e_y})
                epoch_loss += c
            print("the epoch",epoch , "loss is",epoch_loss)
            correct = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print("the accuracy: ",accuracy.eval({x: mnist.test.images , y : mnist.test.labels}))

train_neural_networks(x)


