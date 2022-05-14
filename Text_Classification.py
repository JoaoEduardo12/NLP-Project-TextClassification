import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
#from keras.preprocessing import text
#import tensorflow as tf
import xml.etree.ElementTree as et
#from Machine_Learning import Classifiers
from NLP import TextProcessing
from ML import Classifier

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
    os.mkdir("results")
    train_dic = get_dic(args.TrainLabelFile)
    test_dic = get_dic(args.TestLabelFile)
    for task in train_dic:
        write_report(task = task, method = args.Class, start = True)
        print(f"\n\n TASK: {task}\n\n")
        for disease in test_dic[task]:
            corpus = TextProcessing(train_dic, test_dic, task, args.DataDirectory, disease)            
            X_train, X_test, y_train, y_test = corpus._get_train_test()
            print(f"{disease} classification: \n")
            ml_model = Classifier(X_train, X_test, y_train, y_test, args.Class)
            print(ml_model.metrics)
            write_report(task = task, method = args.Class, disease = disease, metrics = ml_model.metrics)

def write_report(task, method, disease = "", metrics = "", start = False):
    if start:
        file = open(f"results/results_{task}_{method}.txt","a+")
        file.write(f"TASK {task}\n\n")
        file.close()
    else:
        file = open(f"results/results_{task}_{method}.txt","a+")
        bar = f"=====================================================\n"
        file.write(f"\n{disease} classification: \n")
        file.write(bar)
        file.write(metrics + "\n")
        file.write(bar)
        file.close()

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

if __name__ == "__main__":
	args = parse_args()
	main(args)
