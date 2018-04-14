import sys
sys.path.append('C:\python64\Lib\site-packages')
import tensorflow as tf

x1=tf.constant([5 , 6])
x2=tf.constant([7 , 8])
res = x1*x2

#res = tf.multiply(x1,x2)
#res = tf.matmul(x1,x2)

# print(res)
# sess = tf.Session()
# print(sess.run(res))
# sess.close()
# for automatic closing of session
with tf.Session() as ses:
    print(ses.run(res))

