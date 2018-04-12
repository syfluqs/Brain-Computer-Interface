__author__ = "roy"

import numpy as np
from rsw import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# run "export PYTHONWARNINGS=ignore" in shell to ignore FutureWarnings from h5py

import tensorflow as tf

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical


class neural_net:

    class SizeMismatch(Exception):
        def __str__(self):
            return "Size Mismatch in arrays"

    class NonExistentSynapse(Exception):
        def __str__(self):
            return "This synapse does not exist"

    def __init__(self,cluster,layer_sizes,init='randomize'):
        self.cluster = cluster
        self.layer_sizes = layer_sizes
        self._last_err = None

        self._pickler = pickler(cluster)

        if self._pickler.cluster_exists(cluster):
            self.syn = self._pickler.load()
        else:
            # Synapse layers, random weights with zero mean
            self.syn = []
            z = list(self.layer_sizes)+[1]
            for i in range(len(self.layer_sizes)-1):
                if init=='randomize':
                    self.syn.append(2*np.random.random((z[i], z[i+1]))-1)
                elif init=='zeros':
                    self.syn.append(np.zeros((z[i], z[i+1])))
                else:
                    raise NameError("Invalid init parameter (only 'zeros' and 'randomize' accepted)")
            self.write_back()


    def feedforward(self, input_arr):
        if (input_arr.shape[1]==self.layer_sizes[0]):
            L=[input_arr]
            for i in range(len(self.layer_sizes)-1):
                L.append(neural_net.sigmoid(np.dot(L[i],self.syn[i])))
            return L
        else:
            raise neural_net.SizeMismatch

    def backpropagate(self, input_arr, target_arr, learning_rate, reps):
        for M in range(reps):
            delta = []
            # correction = (left_neuron_value) dot (del_sigmoid(right_neoron)*error)
            # delta = (del_sigmoid(right_neoron)*error)
            L = self.feedforward(input_arr)
            # output layer errors
            err_o = (L[-1]-target_arr)
            delta.append(err_o*neural_net.del_sigmoid(L[-1]))
            # subsequent deltas
            for i in range(len(self.layer_sizes)-2,0,-1):
                # subsequent errors = (previous delta) dot (current syn.T)
                delta.append(np.dot(delta[-1],self.syn[i].T)*self.del_sigmoid(L[i]))
            # correction in synapses
            for i in range(len(self.layer_sizes)-2,-1,-1):
                self.syn[i] -= learning_rate*np.dot(L[i].T,delta[len(self.layer_sizes)-2-i])
            if M==reps-1:
                self._last_err = err_o
        # write back to cluster
        # self._pickler.write(self.syn)
        self.write_back()

    def write_back(self):
        self._pickler.write(self.syn)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def del_sigmoid(x):
        return x*(1-x)


class neurel_net_tf:
    ''' Simple feedforward percpetron implementation with Tensorflow
    '''

    def __init__(self,cluster,layer_sizes,learning_rate=0.01,epochs=5):
        self.cluster = cluster
        self.layer_sizes = layer_sizes
        self.avg_err = 0
        self.session = tf.Session()

        self._saver = tf.train.Saver()
        self._simple_save = tf.saved_model.simple_save

        seed = 1234

        if os.path.isfile(cluster):
            # load model from file
            pass
        else:
            # define inputs, outputs, weights and bias weights
            self.x = tf.placeholder((tf.float32), [None, layer_sizes[0]]) # input to neural net
            self.y = tf.placeholder((tf.float32), [None, layer_sizes[-1]]) # output from neural net

            self.weights = []
            for n in (range(len(layer_sizes))-1):
                self.weights.append(tf.Variable(tf.random_normal([layer_sizes[n],layer_sizes[n+1]]),seed = seed))

            self.biases = []
            for n in (range(len(layer_sizes))-1):
                self.biases.append(tf.Variable(tf.random_normal([layer_sizes[n+1]]),seed = seed))

            # define new graph structure
            self.layers = []
            # first layer
            self.layers.append(tf.add(tf.matmul(self.x,weights[0],biases[0])))
            # hidden layers
            for n in (range(1,len(layer_sizes))-2):
                self.layers.append(tf.sigmoid(tf.add(tf.matmul(self.layers[-1],weights[n],biases[n]))))
            # output layer
            self.layers.append(tf.matmul(self.layers[-1],weights[-1])+biases[-1])

            # neural net cost function
            self.cost(tf.reduce_mean(tf.nn_softmax_cross_entropy_with_logits(self.layers[-1],self.y)))

            # neural net optimizer
            self.optimizer = tf.train.AdamOptimzer(learning_rate=learning_rate).minimize(cost)

            # initializing tensor flow variables
            self.init_graph = tf.initialize_all_variables()

            # Start tf session
            self.session.run(self.init_graph)


    def train(self,batch,labels,max_val):
        for e in range(self.epochs):
            self.avg_err = 0
            for i in range(len(batch)):
                _, c = sess.run([self.optimizer, self.cost], feed_dict = {x:batch[i], y:labels[i]})
                self.avg_err += c/len(batch)
            print("Epoch %d avg error = %.5f".format(e+1,self.avg_err))


    def feed(self,input_arr):
        prediction = tf.argmax(self.layers[-1],1)
        output = prediction.eval({x:input_arr.reshape(-1,self.layer_sizes[0])})


class neural_net_tf_dnn:

    def __init__(self,cluster,layer_sizes,learning_rate=0.01):
        self.cluster = cluster
        self.layer_sizes = layer_sizes
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns = my_feature_columns,
            hidden_units = list(layer_sizes)[1:-1],
            n_classes = layer_sizes[-1])

    def train(self):
        self.classifier.train(
            input_fn = lambda:neural_net_tf_dnn.read_trainingset(self.cluster),
            steps=100)

    def evaluate(self):
        eval_result = self.classifier.evaluate(
            input_fn=lambda:neural_net_tf_dnn.read_trainingset(self.cluster))

    # def predict(self,input_arr):
    #     prediction = self.classsifier.predict(
    #         input_fn=


class neural_net_keras:

    def __init__(self,cluster,layer_sizes,learning_rate=0.01):
        self.cluster = cluster
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.training_data_x = None
        self.training_data_y = None

        self.pickler = pickler(cluster)
        if self.pickler.cluster_exists(cluster):
            self.load_training_data()

        if os.path.isfile('data/%s.h5'%(self.cluster)):
            # load model from file
            self.model = load_model('data/%s.h5'%(self.cluster))
        else:
            self.model = Sequential()

            # input and first hiden layer
            self.model.add(Dense(input_dim=self.layer_sizes[0], units=layer_sizes[1], activation='relu', use_bias=True, bias_initializer='zeros'))
            # subsequent hidden layers
            for n in range(1,len(layer_sizes)-2):
                self.model.add(Dense(units=self.layer_sizes[n+1], activation='relu', use_bias=True, bias_initializer='zeros'))
            # output layer
            self.model.add(Dense(units=self.layer_sizes[-1], activation='sigmoid'))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self, batch_size=1, epochs=5):
        x_train, y_train = self.load_training_data()
        self.model.fit(np.array(x_train,"float32"), np.array(y_train,"float32"), epochs=epochs, batch_size=batch_size, verbose=2)
        self.model.save('data/%s.h5'%(self.cluster))

    def train_from_arr(self, x_arr, y_arr, batch_size=None, epochs=5):
        self.model.fit(x_arr, y_arr, epochs=epochs, batch_size=batch_size, verbose=2)
        self.model.save('data/%s.h5'%(self.cluster))

    def evaluate(self, batch_size=1):
        x_test, y_test = self.load_training_data()
        self.model.evaluate(np.array(x_test,"float32"),np.array(y_test,"float32"),batch_size=batch_size)

    def evaluate_from_arr(self, x_arr, y_arr, batch_size=None):
        self.model.evaluate(x_arr,y_arr,batch_size=batch_size)

    def predict(self,x_arr,batch_size=1):
        prediction = self.model.predict(np.array(x_arr,"float32"),batch_size)
        return [x for x in prediction]

    def load_training_data(self):
        if self.pickler.cluster_exists(self.cluster):
            data = self.pickler.load()
            x_d = data['x_train'].tolist() # feature data set
            self.training_data_x = [np.array(x,"float32") for x in x_d]
            y_d = data['y_train'].tolist() # label data set
            self.training_data_y = [np.array(x,"float32") for x in y_d]
            return (self.training_data_x, self.training_data_y)
        else:
            raise Exception("Training data does not exists for cluster \'%s\'"%(self.cluster))

    def add_to_training_data(self,x_arr,y_arr):
        if self.training_data_x == None:
            self.training_data_x = []
        if self.training_data_y == None:
            self.training_data_y = []
        x_arr = np.array(np.squeeze(x_arr),"float32")
        y_arr = np.array(np.squeeze(y_arr),"float32")
        self.training_data_x.append(x_arr)
        self.training_data_y.append(y_arr)

    def write_back_training_data(self):
        self.pickler.write(x_train=self.training_data_x, y_train=self.training_data_y)