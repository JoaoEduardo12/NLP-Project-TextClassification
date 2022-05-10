import sys, os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
#from tensorflow.keras.preprocessing import text
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import xml.etree.ElementTree as et
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def parse_args():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(prog="medical_text_classification")
    parser = argparse.ArgumentParser(description='Given a directory with text files, a file with label indications, and a classifier, this program proceeds to classify text')
    parser.add_argument('-classifier', '--Class', help = 'Give this command, a classification algorithm, including linear_svm, neural_networks, kernel_svm")')
    parser.add_argument('-data_dir', '--DataDirectory', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-process_text', '--ProcessText', help = 'Give this command a file with the search term in NCBI (right side of the page in "Search Details, when you query an entry")')
    parser.add_argument('-train_label_file', '--TrainLabelFile', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-test_label_file', '--TestLabelFile', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-out', '--OutputFile', help = 'Give this command the label files containing either train and test data')
    args = parser.parse_args()
    return args

def main(args):
    train_dic = get_dic(args.TrainLabelFile)
    test_dic = get_dic(args.TestLabelFile)
    for task in train_dic:
        file = open(f"results_{task}_{args.Class}.txt","a+")
        print(f"\n\n TASK: {task}\n\n")
        file.write(f"TASK {task}\n\n")
        for disease in test_dic[task]:
            X_train, y_train = iterate_text(train_dic, task, args.DataDirectory, disease)
            X_test, y_test = iterate_text(test_dic, task, args.DataDirectory, disease)
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)
            print(f"{disease} classification: \n")
            file.write(f"\n{disease} classification: \n")
            file.write(f"=====================================================\n")
            if args.Class == "linear_svm":
                print("Doing SVM")
                clf = perform_svm(X_train, y_train)
            elif args.Class == "logistic_regression":
                print("Doing Logistic Regression")
                clf = perform_lr(X_train, y_train)
            print("Predicting...")
            # Form a prediction set
            predictions = clf.predict(X_test)
            # Print a classification report
            print(metrics.classification_report(y_test,predictions))
            file.write(metrics.classification_report(y_test,predictions) + "\n")
            file.write(f"=====================================================\n")
        file.close()

def perform_svm(train, test):
    clf = LinearSVC()
    clf.fit(train, test)
    return clf

def perform_lr(train, test):
    clf = LogisticRegression()
    clf.fit(train, test)
    return clf

def get_dic(string):
    #get dict from xml
    train_labels = et.parse(string)
    root = train_labels.getroot()
    dic = {}
    for child in root:
        key = child.attrib['source']
        sub_dic = {}
        for sub_child in child:
            sub_key =  sub_child.attrib['name']
            value = []
            for sub_sub_child in sub_child:
                value.append(sub_sub_child.attrib['id']+','+sub_sub_child.attrib['judgment'])
            sub_dic[sub_key]= value
        dic[key] = sub_dic    
    return dic

# nested dictionary
# key: textal and intuitive
# key of keys: diseases
# values of keys of keys: list with "number,label"

def iterate_text(dic1, tag, direc, disease):
    X = []
    y = []
    for elements in dic1[tag][disease]:
        nb = elements.split(",")[0]
        label = elements.split(",")[1]
        bool_value = validate(label, tag)
        if bool_value:
            y.append(label)
            with open(os.path.join(direc,nb+".txt")) as f:
                X.append(f.read())
    return np.array(X), np.array(y)

def validate(label, tag):
    bool_value = False
    if tag == "textual":
        if label == "U" or label == "Y":
            bool_value = True
    elif tag == "intuitive":
        if label == "N" or label == "Y":
            bool_value = True
    return bool_value

def transform_text(array):    
    transformer = TfidfVectorizer()
    transformed = transformer.fit_transform(array)
    return transformed

if __name__ == "__main__":
	args = parse_args()
	main(args)
