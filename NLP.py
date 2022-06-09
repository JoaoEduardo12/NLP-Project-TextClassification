import os
import numpy as np
#from gensim.utils import Word2Vec

class TextProcessing:
	
    def __init__(self, train_dict, test_dict, task, data_dir, disease, adv_bool):
        '''
		Initialize class
		'''
        self.task = task
        self.data_dir = data_dir
        self.disease = disease
        self.X_train, self.y_train, self.X_train_list = self._iterate_text(train_dict)
        self.X_test, self.y_test, self.X_test_list = self._iterate_text(test_dict)
        self.X_train_vect, transformer = self._fit_transform_text(self.X_train)
        self.X_test_vect = self._transform_text(self.X_test, transformer)
        if adv_bool == "True":
            import spacy
            self.nlp = spacy.load("en_core_web_lg")

    def _iterate_text(self, data_dict):
        '''
		This method iterates to the given file by the user and extracts our X and y data
		'''
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
        return np.array(X), np.array(y), X

    def _validate(self, label, tag):
        '''
		This method is important for removing classes with
		very few examples both in the textal and intuitive challenges
		'''
        bool_value = False
        if tag == "textual":
            if label == "U" or label == "Y":
                bool_value = True
        elif tag == "intuitive":
            if label == "N" or label == "Y":
                bool_value = True
        return bool_value

    def _fit_transform_text(self,array):
        '''
		This method performs term frequency inverse document frequency transformation
		to our data so we can feed it later to a ml model
		'''
        from sklearn.feature_extraction.text import TfidfVectorizer
        transformer = TfidfVectorizer(ngram_range = (1,2)) # filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'' min_df=2, stop_words='english', strip_accents='unicode', norm='l2'
        transformed = transformer.fit_transform(array).todense()
        return transformed, transformer

    def _transform_text(self,array,transformer):
        '''
		Its important this method inherits the transformer we got from X_train,
		so we can only use transform() and not fit_transform()
		'''
        return transformer.transform(array).todense()

    def _get_train_test(self):
        '''
		Return all component data
		'''
        return self.X_train_vect, self.X_test_vect, self.y_train, self.y_test

    def _get_nlp_docs(self):
        '''
		Creates doc objects from spacy  in each of the clinical texts
		'''
        print("\nProcessing train and test datasets with SpaCy.\nIt will take a while..")
        self.X_train_doc = [self.nlp(x) for x in self.X_train_list]
        self.X_test_doc = [self.nlp(x) for x in self.X_test_list]

    def _lemmatization(self):
        '''
		Subsitutes each token by its lemma, and also gets rid of ponctuation
		'''
        for document in range(len(self.X_train_doc) - 1):
            self.X_train_doc[document] = [token.lemma_.lower() for token in self.X_train_doc[document] if not token.is_punct]
        for document in range(len(self.X_test_doc) - 1):
            self.X_test_doc[document] = [token.lemma_.lower() for token in self.X_test_doc[document] if not token.is_punct]

    def _add_stop_word_vocab(self, word_set):
        '''
		Adds a specified word to the spacy english vocabolary of stop word
		May come in handy
		'''
        self.nlp.Defaults.stop_words.union(word_set)

    def _remove_stop_word_vocab(self, word):
        '''
		Ditto
		'''
        self.nlp.Defaults.stop_words.remove(word)

    def _remove_stop_words(self):
        '''
		This method removes all stop words from the current documents
		'''
        for document in range(len(self.X_train_doc) - 1):
            self.X_train_doc[document] = [token.lower() for token in self.X_train_doc[document] if not token.lower() in self.nlp.Defaults.stop_words]
        for document in range(len(self.X_test_doc) - 1):
            self.X_test_doc[document] = [token.lower() for token in self.X_test_doc[document] if not token.lower() in self.nlp.Defaults.stop_words]

    def _return_processed_data(self, model):
        '''
		returns the processed text to X_train and X_test
		'''
        self.X_train_final = np.array([str(x) for x in self.X_train_doc])
        self.X_test_final = np.array([str(x) for x in self.X_test_doc])
        if model == "linear_svm" or model == "logistic_regression": 
            self.X_train_final, transformer = self._fit_transform_text(self.X_train_final)
            self.X_test_final = self._transform_text(self.X_test_final, transformer)
        return self.X_train_final, self.X_test_final

    def set_custom_boundaries(self, doc):
        '''
		Not yet implemented
		'''
        for token in doc[:-1]:
            if token.text == ';':
                doc[token.i+1].is_sent_start = True
        return doc

    def _get_vocab_length(self, doc):
        return len(doc.vocab)

    def _remove_sentence(self, doc, nb):
        '''
		Not yet implemented
		'''
        sentence = list(doc.sents)[nb]
    
    def _get_named_entities_length(self):
        return len(self.doc.ents)

    def _set_custom_boundaries(self, string):
        '''
        Adds a custom string that can be used to seperate new sentences in the file
        '''
        for token in self.doc[:-1]:
            if token.text == string:
                self.doc[token.i+1].is_sent_start = True
        return self.doc

    def _get_max_length_docs(self):
        max_x_train = max([len(x) for x in self.X_train_doc])
        max_x_test = max([len(x) for x in self.X_test_doc])
        return max(max_x_train, max_x_test)

    def _clean_data(self):
        import re
        for document in range(len(self.X_train_doc) - 1):
            self.X_train_doc[document] = [re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', sentence) for sentence in self.X_train_doc[document]]
            self.X_train_doc[document] = [x for x in self.X_train_doc[document] if x.isalpha()]
            self.X_train_doc[document] = [x for x in self.X_train_doc[document] if len(x) > 2]
        for document in range(len(self.X_test_doc) - 1):
            self.X_test_doc[document] = [re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', sentence) for sentence in self.X_test_doc[document]]
            self.X_test_doc[document] = [x for x in self.X_test_doc[document] if x.isalpha()]
            self.X_train_doc[document] = [x for x in self.X_train_doc[document] if len(x) > 2]
