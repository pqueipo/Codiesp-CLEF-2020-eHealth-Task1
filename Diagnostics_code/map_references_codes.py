!pip install -U scikit-learn
!pip install sklearn

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#dataframes
import pandas as pd
from tabulate import tabulate
import numpy as np

#regex
import re

#models
import spacy

import re
import ast
import tqdm
import pickle
import collections
import numpy as np
import scipy as sp
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords_en = stopwords.words("english")
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import ast
from ast import literal_eval

#Grader
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2')
import grader
from grader import Grader
grader = Grader()


path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/'

path_T1 = path_codiesp + 'datasetv3/train/train_articles_codiesp.csv'
path_T2 = path_codiesp + 'datasetv3/dev/dev_articles_codiesp.csv'
path_T3 = path_codiesp + 'datasetv3/test/test_articles_codiesp.csv'
path_T4 = path_codiesp + 'datasetv3/train/train_articles_en_codiesp.csv'
path_T5 = path_codiesp + 'datasetv3/dev/dev_articles_en_codiesp.csv'
path_T6 = path_codiesp + 'datasetv3/test/test_articles_en_codiesp.csv'

path_A1 = path_codiesp + 'datasetv3/train/trainD.tsv'
path_A2 = path_codiesp + 'datasetv3/dev/devD.tsv'
path_A3 = path_codiesp + 'datasetv3/train/trainP.tsv'
path_A4 = path_codiesp + 'datasetv3/dev/devP.tsv'

path_B1 = path_codiesp + 'datasetv3/train/mapped_annotations/trainD_annotations_mapped'
path_B2 = path_codiesp + 'datasetv3/dev/mapped_annotations/devD_annotations_mapped'
path_B3 = path_codiesp + 'datasetv3/train/mapped_annotations/trainP_annotations_mapped'
path_B4 = path_codiesp + 'datasetv3/dev/mapped_annotations/devP_annotations_mapped'

path_C1 = path_codiesp + 'codiesp_codesv2/codiesp-D_codes.tsv'
path_C2 = path_codiesp + 'codiesp_codesv2/codiesp-P_codes.tsv'

path_J1 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/articles_annotated/train_annotated_articles_D.csv'
path_J2 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/articles_annotated/dev_annotated_articles_D.csv'
path_J3 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/articles_annotated/train_annotated_articles_en_D.csv'
path_J4 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/articles_annotated/dev_annotated_articles_en_D.csv'

path_J5= '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/articles_annotated/train_annotated_articles_P.csv'
path_J6 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/articles_annotated/dev_annotated_articles_P.csv'
path_J7 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/articles_annotated/train_annotated_articles_en_P.csv'
path_J8 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/articles_annotated/dev_annotated_articles_en_P.csv'

path_K1 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/K_mapped_annotated_articles/trainD_mapped_annotated_articles.csv'
path_K2 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/K_mapped_annotated_articles/devD_mapped_annotated_articles.csv'
path_K3 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/K_mapped_annotated_articles/trainD_mapped_annotated_articles_en.csv'
path_K4 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/K_mapped_annotated_articles/devD_mapped_annotated_articles_en.csv'

path_K5= '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/K_mapped_annotated_articles/trainP_mapped_annotated_articles.csv'
path_K6 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/K_mapped_annotated_articles/devP_mapped_annotated_articles.csv'
path_K7 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/train/K_mapped_annotated_articles/trainP_mapped_annotated_articles_en.csv'
path_K8 = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/K_mapped_annotated_articles/devP_mapped_annotated_articles_en.csv'

def load_articles(path):
  df = pd.read_csv(path, sep=",", engine="python", encoding='utf-8')
  print("--- Loaded dataset:", path)

  print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(df.head())
  #print(tabulate(df.head(), headers='keys', tablefmt='psql'))

  return df

path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/'

print("- Load T1. Articles, Diagnostics/Procedures, train")
train_file_articles = path_codiesp + 'datasetv3/train/train_articles_codiesp.csv'
df_train_articles = load_articles(train_file_articles)

print("- Load T2. Articles, Diagnostics/Procedures, dev")
dev_file_articles = path_codiesp + 'datasetv3/dev/dev_articles_codiesp.csv'
df_dev_articles = load_articles(dev_file_articles)

"""Paths"""
path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp/'
path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp/'
names_codeD=["Code", "Es-description", "En-description"]   # Columns: code \t es-description \t en-description
names_codeP=["Code", "Es-description", "En-description", "Approach?"]   # Columns: code \t es-description \t en-description \t approach?
file_codeD = path_codiesp + 'codiesp_codes/codiesp-D_codes.tsv'


def load_data(path, names):
  df = pd.read_csv(path, sep="\t", names=names)
  #print("--- Loaded dataset:", path)
  #print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(tabulate(df.head(), headers='keys', tablefmt='psql'))
  return df

"""--- Load Code, train, test"""
df_codeD = load_data(file_codeD, names_codeD)
#print(tabulate(df_codeD.head(), headers='keys', tablefmt='psql'))


def load_data(path, names):
  df = pd.read_csv(path, sep="\t", names=names)
  print("--- Loaded dataset:", path)

  print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(df.head())
  #print(tabulate(df.head(), headers='keys', tablefmt='psql'))

  return df

#It uses load_data(path, names)
 
path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/'
names_codeD=["Code", "Es-description", "En-description"]   # Columns: code \t es-description \t en-description
names_codeP=["Code", "Es-description", "En-description"]   # Columns: code \t es-description \t en-description \t approach?

print("- Load C1. Codes, Diagnostics")
file_codeD = path_codiesp + 'codiesp_codesv2/codiesp-D_codes.tsv'
df_codeD = load_data(file_codeD, names_codeD)

def read_data(filename):
    data = pd.read_csv(filename, sep=',', usecols=["article_content", "tags"])
    #print(data['tags'])
    #print(data['article_content'])
    return data


path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/'

print("- Read K3. Joined, Mapped, Diagnostics, train, english")
trainD_en = read_data(path_K3)
print("- Read K4. Joined, Mapped, Diagnostics, dev, english")
devD_en = read_data(path_K4)



print('Shape of data')
print(f'Train: {trainD_en.shape}')
print(f'Validation: {devD_en.shape}')

train = trainD_en
val = devD_en

import collections

def Dict_tag_id_tag_mapping(df_code):
  tags_list = []
  for i in range (len(df_code['Code'])):
    tags_together = df_code['Code'][i]
    tags_together = tags_together.replace(']', '')
    tags_together = tags_together.replace('[', '')
    tags_together = tags_together.replace('\'', '')
    tags_separated = tags_together.split(', ')
    #print(tags_separated)
    for j in tags_separated:
      #print(j)
      tags_list.append(j)	  
  #print(tags_list)
  
  #Converting a list to dictionary with list elements as values in dictionary and keys are enumerated index starting from 0 i.e. index position of element in list
  dictOfTags = { tags_list[i] : i for i in range(0, len(tags_list) ) }
  print(dictOfTags)

  return(dictOfTags)

codeD_dict_tag_id_mapping = Dict_tag_id_tag_mapping(df_codeD)
codeP_dict_tag_id_mapping = Dict_tag_id_tag_mapping(df_codeP)

print(len(codeD_dict_tag_id_mapping))
print(len(codeP_dict_tag_id_mapping))
def Create_transposed_list(list):
  #print("lenght and list are: ", len(list), "\t", list)
  np_list = np.asarray([list]).T
  #print("lenght and transposed token IOB list are: ", len(np_list), "\t", np_list)
  return(np_list)


def Create_mapped_tag_list(df_annotations, code_dict_tag_id_mapping):
  mapped_tag_list=[]
  for i in range(len(df_annotations)):
    t = df_annotations['ICD10-code'][i]    #to add '' : str = "\'" + t +  "\'" 
    upper = t.upper()
    #print(upper)
    mapping = code_dict_tag_id_mapping[upper]
    #print(mapping)
    #print("The code: ", upper, "\thas the mapping: ", mapping)
    mapped_tag_list.append(mapping)

  #Into numpy array
  np_mapped_tag_list =  Create_transposed_list(mapped_tag_list)      ##Nested function
  #print(np_mapped_tag_list)
  return(np_mapped_tag_list)

def Create_mapped_tag_dataframe(df_annotations, code_dict_tag_id_mapping, path):
  id_list = df_annotations['ArticleID']
  np_id_list =  Create_transposed_list(id_list)                      ##Nested function
  
  np_mapped_tag_list = Create_mapped_tag_list(df_annotations, code_dict_tag_id_mapping)        ##Nested function

  df = pd.DataFrame(columns=['article_id','mapped_tags'])
  for a in range(len(id_list)):
    df.loc[a] = [np.random.randint(0,2) for n in range(2)]
  for b in range(len(id_list)):
    df.loc[:, ['article_id']] = np_id_list
    df.loc[:, ['mapped_tags']] = np_mapped_tag_list
  print(tabulate(df.head(), headers= 'keys', tablefmt='psql'))

  df.to_csv(path, index=False)             #guardar dataframe como tsv sin el indice de la izquierda

  return(df)


path = path_codiesp + 'datasetv3/train/mapped_annotations/trainD_annotations_mapped'
print("- Save Mapped A1. Annotations, Diagnostics, train, dataframe in: ", path)
df_trainD_annotations_mapping =  Create_mapped_tag_dataframe(df_train_annotationsD, codeD_dict_tag_id_mapping, path) 

path = path_codiesp + 'datasetv3/dev/mapped_annotations/devD_annotations_mapped'
print("- Save Mapped A2. Annotations, Diagnostics, dev, dataframe in: ", path)
df_devD_annotations_mapping =  Create_mapped_tag_dataframe(df_dev_annotationsD, codeD_dict_tag_id_mapping, path) 

path = path_codiesp + 'datasetv3/train/mapped_annotations/trainP_annotations_mapped'
print("- Save Mapped A4. Annotations, Procedures, dev, dataframe in: ", path)
df_trainP_annotations_mapping =  Create_mapped_tag_dataframe(df_train_annotationsP, codeP_dict_tag_id_mapping, path) 


path = path_codiesp + 'datasetv3/dev/mapped_annotations/devP_annotations_mapped'
print("- Save Mapped A4. Annotations, Procedures, dev, dataframe in: ", path)
df_devP_annotations_mapping =  Create_mapped_tag_dataframe(df_dev_annotationsP, codeP_dict_tag_id_mapping, path) 

print("- Dict A1. Annotations, Diagnostics, train")
df_train_annotationsD

print("- Dict A2. Annotations, Diagnostics, dev")
df_dev_annotationsD

print("- Dict A3. Annotations, Procedures, train")
df_train_annotationsP   

print("- Dict A4. Annotations, Procedures, dev")
df_dev_annotationsP

import collections

def Dict_with_tags_frequency(df_annotations):
  #tag como string
  tag2count = collections.defaultdict(lambda: 0)      #number of times that every tag in the list of tags appeas. default 0
  for tags in df_annotations['ICD10-code']:
      #print(tags)
      for tag in tags.split():
      #for i in range (len(train['tags'])):
          #print(tag)
          #print(type(tag))
          t = tag.replace('\'', '')
          t = t.replace(']', '')
          t = t.replace('[', '')
          t = t.replace(',', '')
          #print(t)
          tag2count[t] += 1
          #print(tag2count[tag])
  most_common_tags = sorted(tag2count.items(),
                            key=lambda x: x[1],
                            reverse=True)

  print(most_common_tags)
  return(tag2count, most_common_tags)

trainD_tag2count, trainD_most_common_tags = Dict_with_tags_frequency(df_train_annotationsD)
devD_tag2count, devD_most_common_tags = Dict_with_tags_frequency(df_dev_annotationsD)
trainP_tag2count, trainP_most_common_tags = Dict_with_tags_frequency(df_train_annotationsP)
devP_tag2count, devP_most_common_tags = Dict_with_tags_frequency(df_dev_annotationsP)

#y_list como string

def Dict_tag_id_tag_mapping(df_annotations):
  tags_list = []
  for i in range (len(df_annotations['ICD10-code'])):
    tags_together = df_annotations['ICD10-code'][i]
    tags_together = tags_together.replace(']', '')
    tags_together = tags_together.replace('[', '')
    tags_together = tags_together.replace('\'', '')
    tags_separated = tags_together.split(', ')
    #print(tags_separated)
    for j in tags_separated:
      #print(j)
      tags_list.append(j)	  
  #print(tags_list)
  
  #Converting a list to dictionary with list elements as values in dictionary and keys are enumerated index starting from 0 i.e. index position of element in list
  dictOfTags = { i : tags_list[i] for i in range(0, len(tags_list) ) }
  print(dictOfTags)

  return(dictOfTags)

trainD_tag_id_mapping = Dict_tag_id_tag_mapping(df_train_annotationsD)
devD_tag_id_mapping = Dict_tag_id_tag_mapping(df_dev_annotationsD)
trainP_tag_id_mapping = Dict_tag_id_tag_mapping(df_train_annotationsP)
devP_tag_id_mapping = Dict_tag_id_tag_mapping(df_dev_annotationsP)

#Converting a list to dictionary with list elements as values in dictionary and keys are enumerated index starting from 0 i.e. index position of element in list

dictOfTags = { i : tags_list[i] for i in range(0, len(tags_list) ) }

dictOfTgs = sorted(dictOfTags)

print(len(dictOfTags))
#nltk.FreqDist(dictOfTags) 

new_tag2count = {}
frequency_list=[]
for k,v in tag2count.items(): 
    #print ("%s -> %s" %(k,v))
    frequency_list.append(v)

new_tag2count = { i : frequency_list[i] for i in range(0, len(frequency_list)) }


