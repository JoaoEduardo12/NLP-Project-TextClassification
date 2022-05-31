import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Conv2D, Embedding, GlobalMaxPooling1D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D, Activation
from keras import regularizers
import numpy as np
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

plt.style.use('ggplot')

class CNN:

    embedding_dim = 200
    seq_length = 15
    num_classes_classes = 2
    num_filters = 256
    kernel_size = 5
    vocab_size = 10000

    hidden_dim = 128

    dropout_keep_prob = 0.8
    learning_rate = 1e-3

    batch_size = 64
    num_epochs = 20

    print_per_batch = 100
    save_per_batch = 10


    def __init__(self, X_train, X_test, y_train, y_test, max_size):
        '''
		Initialize class
		'''
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        lb = preprocessing.LabelBinarizer()
        scaler_min_max = preprocessing.MinMaxScaler(feature_range = (0,1))
        self.X_train = scaler_min_max.fit_transform(X_train.todense())
        self.X_test = scaler_min_max.fit_transform(X_test.todense())
        self.y_train = le.fit_transform(y_train)
        self.y_test = le.transform(y_test)
        self.max_size = max_size
        self.vocab_size = self.X_train.shape[1]
        self.model = self._cnn_architecture()
        self._compile_model()
        self._train_cnn()
        self.y_pred = self._predict()
        self.metrics = self._metrics_scores()


    def _padding(self):
        '''
        Paddes sequences with 0s so they all have the same length
        '''
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        self.X_train = pad_sequences(self.X_train, maxlen = self.max_size, padding = "post")
        self.X_test = pad_sequences(self.X_test, maxlen = self.max_size, padding = "post")

    def _cnn_architecture(self):
        kernel_size = 5
        embedding_dim = 125
        batch_size = 16
        maxlen = 5000
        model = Sequential([
        Embedding(input_dim = self.vocab_size, output_dim = embedding_dim, input_length = maxlen),
        Conv1D(embedding_dim, kernel_size, padding = "same", data_format = "channels_last",kernel_regularizer=regularizers.l2(0.03)),
        Activation("relu"),
        GlobalMaxPooling1D(data_format = "channels_last", keepdims = True),
        Conv1D(embedding_dim-50, kernel_size, padding = "same", data_format = "channels_last",kernel_regularizer=regularizers.l2(0.03)),
        Activation("relu"),
        GlobalMaxPooling1D(data_format = "channels_last"),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
        ])
        print(model.summary())
        return model

    # create embedding matrix using pre-trained word vectors
    def create_embedding_matrix(self):
        vocab, embd, word_vector_map = self.loadWord2Vec("/home/edu/Desktop/Modelacao/StatisticalLearning/NLP/obesity-master/data/mimic3_pp200.txt")
        embedding_dim = len(embd[0])
        sub_embeddings = np.random.uniform(-0.0, 0.0, (self.vocab_size , embedding_dim))
        count = 0
        for i in range(0, self.vocab_size):
            if(self.X_train[i] in word_vector_map): #word_vector_map.has_key(cnn.words[i])
                count = count
                sub_embeddings[i]= word_vector_map.get(self.X_train[i])
            else:
                count = count + 1
        return sub_embeddings, embedding_dim

    def loadWord2Vec(self, filename):
        vocab = []
        embd = []
        word_vector_map = {}
        file = open(filename,'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            if(len(row) > 2):
                vocab.append(row[0])
                embd.append(row[1:])
                word_vector_map[row[0]] = row[1:]
        print('Loaded Word Vectors!')
        file.close()
        return vocab, embd, word_vector_map

    def _compile_model(self):
        #keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=1.), loss = "binary_crossentropy", metrics = ["accuracy"])


    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

    def _train_cnn(self):
        classWeight = compute_class_weight(class_weight = 'balanced', classes = np.unique(self.y_train), y =  np.array(self.y_train))
        classWeight = dict(enumerate(classWeight))
        history = self.model.fit(self.X_train, self.y_train, verbose = 1, epochs = 200, batch_size = 30, validation_data = (self.X_test, self.y_test), class_weight=classWeight) #class_weight=classWeight
        self.plot_history(history)

    def _predict(self):
        return self.model.predict(self.X_test)

    def _metrics_scores(self):
        from sklearn import metrics
        self.y_pred = [1 if x >= 0.5 else 0 for x in self.y_pred]
        return metrics.classification_report(self.y_test, self.y_pred)
