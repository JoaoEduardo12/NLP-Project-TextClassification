import os
import numpy as np
import spacy

class TextProcessing:
	
    def __init__(self, train_dict, test_dict, task, data_dir, disease):
        self.task = task
        self.data_dir = data_dir
        self.disease = disease
        nlp = spacy.load("en_core_web_sm")
        self.X_train, self.y_train = self._iterate_text(train_dict)
        self.X_test, self.y_test = self._iterate_text(test_dict)
        self.X_train_vect, transformer = self._fit_transform_text(self.X_train)
        self.X_test_vect = self._transform_text(self.X_test, transformer)
        #self.X_train_doc = nlp(self.X_train)
        #self.X_test_doc = nlp(self.X_test)

    def _get_train_test(self):
        return self.X_train_vect, self.X_test_vect, self.y_train, self.y_test

    def _get_vocab_length(self):
        return len(self.doc.vocab)
    
    def _get_named_entities_length(self):
        return len(self.doc.ents)

    def _add_stop_word(self, nlp, word):
        return nlp.Defaults.stop_words.add(word)

    def _remove_stop_word(self, nlp, word):
        return nlp.Defaults.stop_words.remove(word)

    def _iterate_text(self, data_dict):
        X = []
        y = []
        for elements in data_dict[self.task][self.disease]:
            nb = elements.split(",")[0]
            label = elements.split(",")[1]
            bool_value = self._validate(label, self.task)
            if bool_value:
                y.append(label)
                with open(os.path.join(self.data_dir,nb+".txt")) as f:
                    X.append(f.read())
        return np.array(X), np.array(y)

    def _validate(self, label, tag):
        bool_value = False
        if tag == "textual":
            if label == "U" or label == "Y":
                bool_value = True
        elif tag == "intuitive":
            if label == "N" or label == "Y":
                bool_value = True
        return bool_value

    def _fit_transform_text(self,array):
        from sklearn.feature_extraction.text import TfidfVectorizer
        transformer = TfidfVectorizer()
        transformed = transformer.fit_transform(array)
        return transformed, transformer

    def _transform_text(self,array,transformer):
        return transformer.transform(array)

    def _set_custom_boundaries(self, string):
        '''
        Adds a custom string that can be used to seperate new sentences in the file
        '''
        for token in self.doc[:-1]:
            if token.text == string:
                self.doc[token.i+1].is_sent_start = True
        return self.doc
