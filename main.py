#This code is used to extract unigram and phrase. Stopwords removed
import os
import sys
import re,string
from nltk.tokenize import word_tokenize
import functools
#import preprocessing
from ND_stopwords import ND_stopwords
from tfidf import get_tf
#from word_count import get_word_count
import json
#from generate_figure import plot_coefficients
#from classifer_svm import linearSVM
from mutiword_phrase import extract_candidate_chunks as ECC

nd_stopwds = ND_stopwords()

def main():
    total = []
    docs = {}
    list_of_docs = []
    y = []
   # seo files
    with open('../item2_files/seo_file_item2_extracted.txt') as f:
     for line in f:
         filename = line[:-1]
         list_of_docs.append(filename)
         print(filename)
         docs[filename] = {'tokens':{}, 'label':{}}
         with open('../item2_files/'+line[:-1]) as f2:
              f2.readline()
              text = f2.readline()
            # remove numbers
              mytext = re.sub(r'\d+(\.\d{1,2})?','',text)

            # get multiword phrase
              multiwords = ECC(mytext)
            # remove stopwords
              multiwords = [w for w in multiwords if w not in nd_stopwds]
            # remove characters with size less than 2 
              multiwords = [w for w in multiwords if len(w) > 2]
            # # remove special characters in a string
              multiwords = [ w.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~=_+''"}) for w in multiwords]

            # get unigrams
              text_words = word_tokenize(mytext)
            # remove punctuation
              text_words = list(filter(lambda x: x not in string.punctuation, text_words))
            # remove stopwords
              words = [w for w in text_words if w not in nd_stopwds]
            # remove characters with size less than 2 and greater than 20
              words = [w for w in words if (len(w) > 2) and (len(w) < 20)]
            # remove special characters in a string
              words = [ w.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~=_+''"}) for w in words]
      
            # join unigram and phrase
              final_tokens = []
              final_tokens.extend(words)
              final_tokens.extend(multiwords)
              clean_word = str.join(',',final_tokens)
              total.append(clean_word)
              y.append(1)
              docs[filename]['tokens'] = clean_word
              docs[filename]['label'] = 1

# non-seo files
    with open('../item2_files_no_seo/seo_non_extracted.txt') as f3:
     for line in f3:
         filename2 = line[:-1]
         list_of_docs.append(filename2)
         print(filename2)
         docs[filename2] = {'tokens':{}, 'label':{}}
         with open('../item2_files_no_seo/'+line[:-1]) as f4:
              f4.readline()
              text2 = f4.readline()
            # remove numbers
              mytext2 = re.sub(r'\d+(\.\d{1,2})?','',text2)

            # get multiword phrase
              multiwords2 = ECC(mytext2)
            # remove stopwords
              multiwords2 = [w for w in multiwords2 if w not in nd_stopwds]
            # remove characters with size less than 2
              multiwords2 = [w for w in multiwords2 if len(w) > 2]
            # # remove special characters in a string
              multiwords2 = [ w.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~=_+''"}) for w in multiwords2]
            
            # get unigrams
              text_words2 = word_tokenize(mytext2)
            # remove punctuation
              text_words2 = list(filter(lambda x: x not in string.punctuation, text_words2))
            # remove stopwords
              words2 = [w for w in text_words2 if w not in nd_stopwds]
            # remove characters with size less than 2 and greater than 20 
              words2 = [w for w in words2 if (len(w) > 2) and (len(w) < 20)]
            # remove special characters in a string
              words2 = [ w.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~=_+''"}) for w in words2]

              final_tokens2 = []
              final_tokens2.extend(words2)
              final_tokens2.extend(multiwords2)
              clean_word2 = str.join(',',final_tokens2)
              total.append(clean_word2)
              y.append(0)
              docs[filename2]['tokens'] = clean_word2
              docs[filename2]['label'] = 0
    
  # write docs doc_of_list to file
    with open('final_tokens_all_with_label_len2.txt','w') as f5:
         json.dump(docs,f5)
    with open('doc_of_list.txt','w') as f6:
         json.dump(list_of_docs,f6)

#    linearSVM(total,y) 

if __name__ == "__main__":
    main()
