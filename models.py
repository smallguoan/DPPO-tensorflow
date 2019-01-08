import tensorflow as tf
import numpy as np


class Models(object):
    def __init__(self,action_num):
        # define all placeholders we should have
        self.s=tf.placeholder(tf.float32,shape=[None,84,84,4],name="state")
        self.R=tf.placeholder(tf.float32,shape=[None,1],name="Reward")
        self.V_s_=tf.placeholder(tf.float32,shape=[None,1],name="V_s_")
        self.acts=tf.placeholder(tf.int32,shape=[None,1],name="actions")
        self.advantages=tf.placeholder(tf.float32,shape=[None,1],name="advantages")
        self.returns=tf.placeholder(tf.float32,shape=[None,1],name="Returns")
        self.action_num = action_num

        #Build pi and old pi
        self.pi,self.value,self.pi_params=self._build_net("Pi",Trainable=True)
        self.oldpi,self.oldvalue,self.oldpi_params=self._build_net("OldPi",Trainable=False)


    def _build_net(self,name,Trainable):
        with tf.variable_scope(name):

            X = tf.to_float(self.s) / 255.0
            batch_size = tf.shape(self.s)[0]


            conv1 = tf.layers.conv2d(
                X, 32, 8, 4, activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),trainable=Trainable)
            conv2 = tf.layers.conv2d(
                conv1, 64, 4, 2, activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),trainable=Trainable)
            conv3 = tf.layers.conv2d(
                conv2, 64, 3, 1, activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),trainable=Trainable)

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.layers.dense(flattened, 512,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),activation=tf.nn.relu,trainable=Trainable)
            prob = tf.layers.dense(fc1, self.action_num,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),activation=None,trainable=Trainable)
            value=tf.layers.dense(fc1,1,kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2.)),activation=None,trainable=Trainable)


        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return prob,value,params

