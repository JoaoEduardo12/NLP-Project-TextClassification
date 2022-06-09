import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Conv2D, Embedding, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D, Activation, Input
from keras import regularizers
import numpy as np
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical
from keras.metrics import AUC
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import text
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
plt.style.use('ggplot')

from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

#### if use tensorflow=2.0.0, then import tensorflow.keras.model_selection 
from tensorflow.keras import backend as K

class CNN:


    def __init__(self, X_train, X_test, y_train, y_test, max_size):
        '''
		Initialize class
		'''
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        lb = preprocessing.LabelBinarizer()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = le.fit_transform(y_train)
        self.y_test = le.transform(y_test)
        self.max_size = max_size
        self.max_size = 2000
        self.vocab_size = 2000
        tokenizer = text.Tokenizer(num_words = 2000, oov_token="<00V>")
        tokenizer.fit_on_texts(X_train)
        self.X_train = tokenizer.texts_to_sequences(X_train)
        self.X_test = tokenizer.texts_to_sequences(X_test)
        self.embedding_matrix = self.create_embedding_matrix(tokenizer)
        self._padding()
        self.model = self.cnn_text_classification()
        self._train_cnn()
        self.y_pred = self._predict()
        self.metrics = self._metrics_scores()


    def _padding(self):
        '''
        Paddes sequences with 0s so they all have the same length
        '''
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        self.X_train = pad_sequences(self.X_train, maxlen = self.max_size, padding = "post", truncating = "post")
        self.X_test = pad_sequences(self.X_test, maxlen = self.max_size, padding = "post", truncating = "post")

    def cnn_text_classification(self):
        # channel 1
        vocab_size = self.vocab_size
        length = self.max_size
        #vocab_size = 2000
        #length = 2000
        inputs = Input(shape=(length,))
        #embedding1 = Embedding(vocab_size, 182)(inputs)
        embedding1 = Embedding(vocab_size, 200, weights = [self.embedding_matrix], trainable = False)(inputs)
        conv1 = Conv1D(filters=14, kernel_size=3, activation='relu', padding = "valid", kernel_regularizer=regularizers.l2(3))(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D()(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        embedding2 = Embedding(vocab_size, 200)(inputs)
        conv2 = Conv1D(filters=14, kernel_size=4, activation='relu', padding = "valid", kernel_regularizer=regularizers.l2(3))(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D()(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        embedding3 = Embedding(vocab_size, 200)(inputs)
        conv3 = Conv1D(filters=14, kernel_size=5, activation='relu', padding = "valid", kernel_regularizer=regularizers.l2(3))(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D()(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        #dense3 = Dense(200, activation='leaky_relu')(merged)
        #drop4 = Dropout(0.5)(dense3)
        dense1 = Dense(128, activation='relu')(merged)
        drop6 = Dropout(0.5)(dense1)
        outputs = Dense(1, activation='sigmoid')(drop6)
        model = Model(inputs=[inputs], outputs=outputs)
        # compile
        model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])
        # summarize
        print(model.summary())
        plot_model(model, show_shapes=True, to_file='multichannel.png')
        return model

    def create_embedding_matrix(self, tokenizer):
        '''
        create embedding matrix using pre-trained word vectors
        '''
        embedding_index = dict()
        f = open("obesity-master/data/mimic3_pp200.txt")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = "float32")
            embedding_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.max_size,200))
        for word, index in tokenizer.word_index.items():
            if index > self.max_size - 1:
                break
            else:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def loadWord2Vec(self, filename):
        '''
        Loads word2vec vectors
        '''
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
        self.model.compile(optimizer = keras.optimizers.SGD(learning_rate=1e-2), loss = "binary_crossentropy", metrics = ["accuracy"])

    def macro_f1(self,y_true, y_pred):    
        def recall_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            
            recall = TP / (Positives+K.epsilon())    
            return recall 
        
        
        def precision_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        
            precision = TP / (Pred_Positives+K.epsilon())
            return precision 
        
        precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
        
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
        history = self.model.fit(self.X_train, self.y_train, verbose = 1, epochs = 40, batch_size = 50, validation_data = (self.X_test, self.y_test), class_weight=classWeight) #class_weight=classWeight
        #self.plot_history(history)

    def _predict(self):
        return self.model.predict(self.X_test)

    def _metrics_scores(self):
        from sklearn import metrics
        self.y_pred = [1 if x >= 0.5 else 0 for x in self.y_pred]
        return metrics.classification_report(self.y_test, self.y_pred)
