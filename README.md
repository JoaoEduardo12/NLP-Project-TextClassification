# NLP-Project-TextClassification
Using natural language processing tools alongside Deep Learning models to predict present diseases in clinical notes of the "obesity" data set

Clinical Text Classification using NLP and Deep Learning

The scripts programmed in this project are for classifying clinical notes in the i2b2 obesity challenge of 2008 described in Yao et al. 2019 (https://doi.org/10.1186/s12911-019-0781-4). This is
a multilabel classification problem, with the goal of labeling 16 diseases as being present, not present, or questionable, in the clinical notes.

I use NLP techniques to process the text, and use SVM, logistic regression, and an implementation of convolutional neural networks (CNNs) to
predict the possible labels. The results I found were that implementing CNNs using 3 channels using allocated pre-trained word embeddings from MIMIC-III
data base, with different kernel sizes in each convolutional
layer works best, though more testing is required.

**Inputs:**

- a name of a classifier, in this case so far I got the options: linear_svm, logistic_regression, and deep_learning;
- the directory where the clinical documents are;
- boolean value detailing whether or not to process text first;
- location of the output file.

**Outputs:**

- directory where the classification results for the method used is stored.

**Requirements:**

- spaCy;
- Scikit-learn;
- Numpy;
- Keras;
- TensorFlow;
- Matplotlib.

**Example use:**

python3 Text_Classification.py -train_label_file 'files/train_groundtruth.xml' -test_label_file 'files/test_groundtruth.xml' -classifier deep_learning -data_dir 'files/obesity_records_no_fam_cuis' -process_text True
