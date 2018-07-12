import json
import numpy as np
import time
#from classifer_svm import linearSVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy.sparse import csr_matrix
from sklearn.utils import resample
#from generate_figure import plot_coefficients
#import matplotlib.pyplot as plt


filename = 'final_tokens_all_with_label_unigrams_pos.txt'
with open (filename) as f:
     data1 = json.load(f)

file2 = 'doc_of_list_unigrams_pos.txt'
with open (file2) as f2:
     list_of_doc_pos = json.load(f2)

total = []
y = []
pos = []
neg = []
y_pos = []

for doc in list_of_doc_pos:
       pos.append(data1[doc]['tokens'])
       y_pos.append(data1[doc]['label'])

filename2 = 'final_tokens_all_with_label_unigrams_neg.txt'
with open (filename2) as f3:
     data2 = json.load(f3)

file4 = 'doc_of_list_unigrams_neg.txt'
with open (file4) as f4:
     list_of_doc_neg = json.load(f4)

for doc in list_of_doc_neg:
       neg.append(data2[doc]['tokens'])

print(len(pos))
print(len(neg))
neg_upsampled = resample(neg,replace=True,n_samples=len(pos)-len(neg),random_state=123)
neg.extend(neg_upsampled)
print(len(neg))
total.extend(pos)
total.extend(neg)
print(len(total))
y_upsampled = [0 for i in range(len(neg))]

y.extend(y_pos)
y.extend(y_upsampled)

tokenize = lambda doc: doc.lower().split(",")
vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
X = vectorizer.fit_transform(total)
X.toarray()
#transformer = TfidfTransformer(smooth_idf=False)
#tfidf = transformer.fit_transform(X)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialise the classifier
depth_options = [1,10,50,100]
for n in depth_options:
    tstart = time.process_time()
    classifier = RandomForestClassifier(n_estimators=200,max_depth=n,min_samples_leaf=5,max_features='sqrt',min_samples_split=2,random_state=0,criterion='entropy')

# train the classifier
    classifier.fit(X_train, y_train)
   
    accuracy = cross_val_score(classifier,X_test,y_test,cv=5,scoring='accuracy')
    precisions = cross_val_score(classifier,X_test,y_test,cv=5,scoring='precision')
    recalls = cross_val_score(classifier,X_test,y_test,cv=5,scoring='recall')
    roc_auc = cross_val_score(classifier,X_test,y_test,cv=5,scoring='roc_auc')
    print('Accuracy:',np.mean(accuracy))
    print('Precision:',np.mean(precisions))
    print('Recall:',np.mean(recalls))
    print('ROC_AUC',np.mean(roc_auc))

    tend = time.process_time()
    print('time used:',tend-tstart)
    print('\n')
