import sys

class Classifier:

    def __init__(self, X_train, X_test, y_train, y_test, model_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        if model_name == "logistic_regression":
            self.clf = self.logistic_regression()
        elif model_name == "linear_svm":
            self.clf = self.linear_svm()
        else: sys.exit("No classifier specified")
        self.y_pred = self._predict()
        self.metrics = self._metrics_scores()

    def logistic_regression(self):
        print("\n Performing Linear SVM \n")
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(self.X_train, self.y_train)
        return clf

    def linear_svm(self):
        print("\n Performing Linear SVM \n")
        from sklearn.svm import LinearSVC
        clf = LinearSVC()
        clf.fit(self.X_train, self.y_train)
        return clf

    def _standardize(self, dataset):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(dataset)
        self.data_z = self.scaler.transform(self.dataset)

    def _cross_validation(self, method):
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import StratifiedKFold
        # from sklearn.model_selection import ShuffleSplit
        # cv = ShuffleSplit(n_splits=4, test_size=0.3) # cross validation normal
        skf = StratifiedKFold(5, shuffle=True)
        scores = cross_val_score(method, self.data_z, self.output, cv=skf, scoring='f1_weighted')
        print(scores)
        return skf

    def _predict(self):
        return self.clf.predict(self.X_test)

    def _metrics_scores(self):
        from sklearn import metrics
        return metrics.classification_report(self.y_test, self.y_pred)

    def _score_metrics(self, method):
        from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
        print(matthews_corrcoef(self.y_test, method.predict(self.X_test)))
        print(f1_score(self.y_test, method.predict(self.X_test), average='binary'))
        print(precision_score(self.y_test, method.predict(self.X_test)))
        print(recall_score(self.y_test, method.predict(self.X_test)))
