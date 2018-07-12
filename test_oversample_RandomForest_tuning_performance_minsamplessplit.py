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


filename = 'final_tokens_all_with_label_unigrams.txt'
with open (filename) as f:
     data = json.load(f)

file2 = 'doc_of_list_unigrams.txt'
with open (file2) as f2:
     list_of_doc = json.load(f2)

total = []
y = []
pos = []
neg = []
y_pos = []

for doc in list_of_doc:
    if data[doc]['label'] == 1:
       pos.append(data[doc]['tokens'])
       y_pos.append(data[doc]['label'])
    else:
       neg.append(data[doc]['tokens'])
       #y_neg.append(data[doc]['label'])

neg_upsampled = resample(neg,replace=True,n_samples=1859,random_state=123)
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
samples_split_options = [2,3,4,5,6,8,10]
for n in samples_split_options:
    tstart = time.process_time()
    classifier = RandomForestClassifier(n_estimators=100,max_depth=50,min_samples_leaf=1,max_features=200,min_samples_split=n,random_state=0,criterion='entropy')

# train the classifier
    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)
    precisions = cross_val_score(classifier,X_test,y_test,cv=5,scoring='precision')
    recalls = cross_val_score(classifier,X_test,y_test,cv=5,scoring='recall')
    print('min_samples_split=',n)
    print('Accuracy:',accuracy_score(y_test,preds))
    print('Precision:',np.mean(precisions))
    print('Recall:',np.mean(recalls))

    false_positive_rate,recall, thresholds = roc_curve(y_test,preds)
    roc_auc = auc(false_positive_rate,recall)
    tend = time.process_time()
    print(tend)
    print('time used:',tend-tstart)
    print('\n')
#plt.plot(false_positive_rate,recall,'b',label='AUC = %0.2f' %roc_auc)
#plt.show()
