import os
import shutil
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xml.etree.ElementTree as et
from NLP import TextProcessing
from ML import Classifier
from DL import CNN

def parse_args():
    '''
    Parsing command line arguments
    '''
    parser = argparse.ArgumentParser(prog="medical_text_classification")
    parser = argparse.ArgumentParser(description='Given a directory with text files, a file with label indications, and a classifier, this program proceeds to classify text')
    parser.add_argument('-classifier', '--Class', help = 'Give this command, a classification algorithm, including linear_svm, deep_learning, logistic_regression")')
    parser.add_argument('-data_dir', '--DataDirectory', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-process_text', '--AdvProc', help = 'Give this command True or False, depending whether or not you would like to perform some NLP tasks before classification")')
    parser.add_argument('-train_label_file', '--TrainLabelFile', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-test_label_file', '--TestLabelFile', help = 'Give this command the label files containing either train and test data')
    parser.add_argument('-out', '--OutputFile', help = 'Give this command the label files containing either train and test data')
    args = parser.parse_args()
    return args

def main(args):
    '''
    Main script
    '''
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")
    train_dic = get_dic(args.TrainLabelFile)
    test_dic = get_dic(args.TestLabelFile)
    for task in train_dic:
        write_report(task = task, method = args.Class, start = True)
        print(f"\n\n TASK: {task}\n\n")
        for disease in test_dic[task]:
            corpus = TextProcessing(train_dic, test_dic, task, args.DataDirectory, disease, args.AdvProc)            
            X_train, X_test, y_train, y_test = corpus._get_train_test()
            if args.AdvProc == "True":
                X_train, X_test = process_text_options(corpus, args.Class)
            print(f"{disease} classification: \n")
            if args.Class == "deep_learning":
                ml_model = CNN(X_train, X_test, y_train, y_test, corpus._get_max_length_docs())
                print(ml_model.metrics)
                write_report(task = task, method = args.Class, disease = disease, metrics = ml_model.metrics)
            else: 
                ml_model = Classifier(X_train, X_test, y_train, y_test, args.Class)
                print(ml_model.metrics)
                write_report(task = task, method = args.Class, disease = disease, metrics = ml_model.metrics)

def process_text_options(corpus, model):
    '''
    This function only runs when process_text is True
    '''
    corpus._add_stop_word_vocab({"'s","puo","i","ii","iii"})
    #corpus._remove_stop_word_vocab("beyond")
    corpus._get_nlp_docs()
    corpus._lemmatization()
    corpus._remove_stop_words()
    corpus._clean_data()
    X_train, X_test = corpus._return_processed_data(model)
    return X_train, X_test

def write_report(task, method, disease = "", metrics = "", start = False):
    '''
    Function to write all the results
    '''
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
    '''
    Get train and test data dictionary from xml files
    '''
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
