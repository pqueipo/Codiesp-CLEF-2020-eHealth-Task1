!pip install spacy-lookup    #For Dictionary based NER ,  Successfully installed flashtext-2.7 spacy-lookup-0.1.0

"""Reference: Github of Isabel Segura Bedmar https://github.com/isegura/BasicNLP
Libraries: 
Spacy https://spacy.io
Spacy-lookup  https://github.com/mpuig/spacy-lookup"""

import pandas as pd
from tabulate import tabulate
import re
import csv
import collections

!pip3 install fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import spacy
from spacy_lookup import Entity
!pip install -U spacy #spacy in /usr/local/lib/python3.6/dist-packages (2.2.4)   1 time
!pip install spacy
!python -m spacy download es          #Every time it reload    ,   Successfully installed es-core-news-sm-2.2.5
!pip install spacy-lookup    #For Dictionary based NER ,  Successfully installed flashtext-2.7 spacy-lookup-0.1.0

nlp = spacy.load('es')          #"en_core_web_sm"
print('spacy.es loaded')

def load_articles(path):
  df = pd.read_csv(path, sep=",", engine="python", encoding='utf-8')
  #print("--- Loaded dataset:", path)
  #print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(tabulate(df.head(), headers='keys', tablefmt='psql'))
  return df

def load_data(path, names):
  df = pd.read_csv(path, sep="\t", names=names)
  #print("--- Loaded dataset:", path)
  #print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(tabulate(df.head(), headers='keys', tablefmt='psql'))
  return df




"""Paths"""
localpath = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD'
#path_codiesp='/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp/'
train_file_articles = localpath + '/final_dataset_v3_to_publish/train/train_articles_codiesp.csv'
dev_file_articles = localpath + '/final_dataset_v3_to_publish/dev/dev_articles_codiesp.csv'
test_file_articles = localpath + '/final_dataset_v3_to_publish/test/test_articles_codiesp.csv'

names_codeD=["Code", "Es-description", "En-description"]   # Columns: code \t es-description \t en-description
names_codeP=["Code", "Es-description", "En-description", "Approach?"]   # Columns: code \t es-description \t en-description \t approach?
file_codeD = localpath + '/codiesp_codes/codiesp-D_codes.tsv'



"""- Load Articles train"""
df_train_articles = load_articles(train_file_articles)
#print(tabulate(df_train_articles.head(), headers='keys', tablefmt='psql'))
df_dev_articles = load_articles(dev_file_articles)
#print(tabulate(df_dev_articles.head(), headers='keys', tablefmt='psql'))
df_test_articles = load_articles(test_file_articles)
#print(tabulate(df_test_articles.head(), headers='keys', tablefmt='psql'))

"""--- Load Code, train, test"""
df_codeD = load_data(file_codeD, names_codeD)
#print(tabulate(df_codeD.head(), headers='keys', tablefmt='psql'))

def ner(text):
  text = text.lower() #en el caso de poner todo minusculas
  document = nlp(text)

  result = []
  for entity in document.ents:                        #divide document into entities
      #print('Type: {}, Value: {}, star: {}, end: {}'.format(entity.label_, entity.text,entity.start_char, entity.end_char))   #start_char and end_char are the position of the first and last character of an entity, respect the document begining.
      result.append(entity.text)   #start_char and end_char are the position of the first and last character of an entity, respect the document begining.
      #print(result)

  return(result)


def load_dictionary(df, language_index):
  """This function reads a df and save the fourth column into a list. 
  This list is an input parameter, which is modified. We need to pass
  the list as parameter, because if the list is returned, 
  its type becomes to NoneType"""
  dictionary=[]
  for index, row in df.iterrows():                                  
    #print(row[1])
    dictionary.append(row[language_index].lower())          #language_index : 1 spanish and 2 english
  #print(dictionary)
  #print('length of the dictionary loaded: ', len(dictionary))
  return(dictionary)



def Dict_code_reference(df):    #Make a dictionary {'A00.0': 'Cólera debido a Vibrio cholerae 01, biotipo cholerae ', 'A00.1': 'Cólera debido a Vibrio cholerae 01, biotipo El Tor',...
  code_list = []
  reference_list = []

  for index, row in df.iterrows():
    code_list.append(row[0])
    reference_list.append(row[1])    #1 spanish, 2 english
  #print(code_list, reference_list)
  
  #Converting a list to dictionary with list elements as values in dictionary and keys are enumerated index starting from 0 i.e. index position of element in list
  dict_code_reference = { code_list[i] : reference_list[i] for i in range(0, len(code_list)) }
  #print(dict_code_reference)

  return(dict_code_reference)



##Spanish

print("---Load dictionary spanish")
diagnostics_dictionary = load_dictionary(df_codeD, language_index=1)

#Add Named Entities metadata to Doc objects in Spacy. First, we load the model and replace the NER module with de entity diagnosticEnt. We do this to avoid overlapping of entities. Then, we also add procedureEnt.
diagnosticEnt = Entity(keywords_list=diagnostics_dictionary,label="DIAGNOSTICO")  #Detect Named Entities using dictionaries. We can process a text and show its entities.


nlp.replace_pipe("ner", diagnosticEnt)   #We replace the common entities with diagnostics
print('entities loaded in nlp')
"""
##English

print("---Load dictionary english")
diagnostics_dictionary = load_dictionary(df_codeD, language_index=2)

#Add Named Entities metadata to Doc objects in Spacy. First, we load the model and replace the NER module with de entity diagnosticEnt. We do this to avoid overlapping of entities. Then, we also add procedureEnt.
diagnosticEnt = Entity(keywords_list=diagnostics_dictionary,label="DIAGNOSTICO")  #Detect Named Entities using dictionaries. We can process a text and show its entities.


nlp.replace_pipe("ner", diagnosticEnt)   #We replace the common entities with diagnostics
print('entities loaded in nlp')
"""

dict_code_reference = Dict_code_reference(df_codeD)

def Find_code_preditions(df, dictionary, path):
  with open(path, 'wt') as out_file:
      tsv_writer = csv.writer(out_file, delimiter='\t')
      for index, row in df.iterrows():
          abs_id = row[0]
          text = row[1]
          result = ner(text)
          length = len(result)
          if (length >0):                                 #no empty text_files
            for i in range(length):
              entity = result[i]
              min_score = 50                              #score ranges
              max_score = -1
              for code, reference in dict_code_reference.items():    
                score = fuzz.ratio(entity, reference)
                if (score > min_score)&(score> max_score):
                  max_reference = reference
                  max_score = score
                  max_code = code
              
              #print(abs_id, max_code, max_score, max_reference, entity)
              #print("----", abs_id, score, result[i]) 
              tsv_writer.writerow([abs_id, max_code, max_score, max_reference, entity])





#Find_code_preditions(df_train_articles, dict_code_reference, localpath +'/predictions/trainD_prediction_codes.tsv')
#Find_code_preditions(df_dev_articles, dict_code_reference, localpath +'/predictions/devD_prediction_codes.tsv')
Find_code_preditions(df_test_articles, dict_code_reference, localpath +'/predictions/testD_prediction_codes.tsv')

df_train_prediction = pd.read_csv(localpath +'/predictions/trainD_prediction.tsv', sep="\t", header=None)
df_dev_prediction = pd.read_csv(localpath +'/predictions/devD_prediction.tsv', sep="\t", header=None)
df_test_prediction = pd.read_csv(localpath +'/predictions/testD_prediction.tsv', sep="\t", header=None)

import spacy
from spacy.matcher import PhraseMatcher
nlp_blank = spacy.blank('en')

drug_list = diagnostics_dictionary
matcher = PhraseMatcher(nlp_blank.vocab)
matcher.add('DRUG', None, *[nlp_blank(entity_i) for entity_i in drug_list])


doc = nlp_blank("dolores de cabeza con brucella")
matches = matcher(doc)

for m_id, start, end in matches:
    entity = doc[start : end] 
    print((entity.text, entity.start_char, entity.end_char, nlp_blank.vocab.strings[m_id]))


from fuzzywuzzy import fuzz
from fuzzywuzzy import process


text = "A patient was prescribed Adepend 5mg, Alfuzosin 20ml and co-magaldrox 5 mg"

for query in drug_list:
    print(process.extractOne(query, text.split()))


