#dataframes
import pandas as pd
from tabulate import tabulate

#regex
import re

def extract_text_files(list):
  #print(list)
  df = pd.DataFrame(data=list)
  texts=[]
  for text in list:      
    text_divided = (text.split())  #separate two elements by /s
    for elements in text_divided:
      texts.append(elements)   #join all the elements in a list
  print(texts)  
  return(texts)

def list_content_text_file(file_name_list, path):
  files_content = []
  for element in file_name_list:
    path_element = path + element   
    #print(element) #name of a file_name
    content_element = open(path_element).read()
    files_content.append(content_element)
    #print(content_element)
  #print(files_content)
  return files_content

def df_text_file(articles_id, articles_content, dataset_path):
   # Calling DataFrame constructor after zipping both lists, with columns specified 
  df = pd.DataFrame(list(zip(articles_id, articles_content)), 
               columns =['article_id', 'article_content']) 

  print("--- Loaded dataset:", dataset_path)
  print("--- Number of rows is " + str(df.shape[0]) + " x number of columns " + str(df.shape[1]))
  #print(df.head())
  print(tabulate(df.head(), headers='keys', tablefmt='psql'))
  return df

def cut_textfile_name(text_list):    
  article_id_list = []
  for file_name in text_list:
    article_id_list.append(file_name.rstrip(".txt"))
  return article_id_list

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD/final_dataset_v3_to_publish/train/text_files'

train_text_file_list = !ls         #Copy all the files

train_text_list = extract_text_files(train_text_file_list)
train_path_text_list = localpath + '/train/text_files/'
train_articles_content = list_content_text_file(train_text_list, train_path_text_list)
train_article_id_list= cut_textfile_name(train_text_list)    #It erases .txt from the text files names in order to obtain articleID
#print(train_article_id_list)

print("- Load Text Files")
df_train_textfiles = df_text_file(train_article_id_list, train_articles_content, train_path_text_list)

df_train_textfiles.to_csv(localpath + '/train/train_articles_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD/final_dataset_v3_to_publish/dev/text_files'

dev_text_file_list = !ls         #Copy all the files

dev_text_list = extract_text_files(dev_text_file_list)
dev_path_text_list = localpath + '/dev/text_files/'
dev_articles_content = list_content_text_file(dev_text_list, dev_path_text_list)
dev_article_id_list= cut_textfile_name(dev_text_list)    #It erases .txt from the text files names in order to obtain articleID
#print(dev_article_id_list)

print("- Load Text Files")
df_dev_textfiles = df_text_file(dev_article_id_list, dev_articles_content, dev_path_text_list)

df_dev_textfiles.to_csv('/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD/final_dataset_v3_to_publish/dev/dev_articles_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD/final_dataset_v3_to_publish/test/text_files'

test_text_file_list = !ls         #Copy all the files

test_text_list = extract_text_files(test_text_file_list)
test_path_text_list = localpath + '/test/text_files/'
test_articles_content = list_content_text_file(test_text_list, test_path_text_list)
test_article_id_list= cut_textfile_name(test_text_list)    #It erases .txt from the text files names in order to obtain articleID
#print(test_article_id_list)

print("- Load Text Files")
df_test_textfiles = df_text_file(test_article_id_list, test_articles_content, test_path_text_list)

df_test_textfiles.to_csv(localpath + '/test/test_articles_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiesp_code/SubtrackD/final_dataset_v3_to_publish/train/text_files_en'

train_text_file_en_list = !ls         #Copy all the files

train_text_en_list = extract_text_files(train_text_file_en_list)
train_path_text_en_list = localpath +'/train/text_files_en/'
train_articles_en_content = list_content_text_file(train_text_en_list, train_path_text_en_list)
train_article_id_en_list= cut_textfile_name(train_text_en_list)    #It erases .txt from the text files names in order to obtain articleID
#print(train_article_id_en_list)

print("- Load Text Files")
df_train_textfiles_en = df_text_file(train_article_id_en_list, train_articles_en_content, train_path_text_en_list)

df_train_textfiles_en.to_csv(localpath +'/train/train_articles_en_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/text_files_en'

dev_text_file_en_list = !ls         #Copy all the files

dev_text_en_list = extract_text_files(dev_text_file_en_list)
dev_path_text_en_list = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/text_files_en/'
dev_articles_en_content = list_content_text_file(dev_text_en_list, dev_path_text_en_list)
dev_article_id_en_list= cut_textfile_name(dev_text_en_list)    #It erases .txt from the text files names in order to obtain articleID
#print(dev_article_id_en_list)

print("- Load Text Files")
df_dev_textfiles_en = df_text_file(dev_article_id_en_list, dev_articles_en_content, dev_path_text_en_list)

df_dev_textfiles_en.to_csv('/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/dev/dev_articles_en_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda

cd '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/test/text_files_en'

test_text_file_en_list = !ls         #Copy all the files

test_text_en_list = extract_text_files(test_text_file_en_list)
test_path_text_en_list = '/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/test/text_files_en/'
test_articles_en_content = list_content_text_file(test_text_en_list, test_path_text_en_list)
test_article_id_en_list= cut_textfile_name(test_text_en_list)    #It erases .txt from the text files names in order to obtain articleID
#print(test_article_id_en_list)

print("- Load Text Files")
df_test_textfiles_en = df_text_file(test_article_id_en_list, test_articles_en_content, test_path_text_en_list)

df_train_textfiles_en.to_csv('/content/drive/My Drive/Colab Notebooks/3. Codiesp/Codiespv2/datasetv3/test/test_articles_en_codiesp.csv', index=False) #guardar dataframe como tsv sin el indice de la izquierda


