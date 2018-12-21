# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:39:48 2018

@author: HP
"""

########## 1) lecture des données ###############
import pandas as pd
Data= pd.read_csv('Tobacco3482.csv')
print(Data[:10])


########### 2) Affichage des données par label: ############
import seaborn as sns

sns.countplot(data=Data,y='label')


##########  attribution de  chaque texte a son label#######
Data_txt=[]
Data_txt=Data

NB = Data_txt.shape[0]
for i in range (NB):
    A = Data_txt.get_value(i, 'img_path')
    Data_txt.set_value(i, 'img_path', 'Tobacco3482-OCR/'+A)
    Data_txt.set_value(i, 'img_path', Data_txt.get_value(i, 'img_path').split('.jpg')[0]+'.txt')
    Data_txt.set_value(i, 'img_path',open(Data_txt.get_value(i, 'img_path'), "r",encoding="utf8").read())
    
####### attribuer le texte a son label(suite ) ########
Data_txt.columns = ['text','label']
print(Data_txt.head())


Data_txt.iloc[11].text



####### diviser les données ##################
from sklearn.model_selection import train_test_split

(X_train,X_app,y_train,y_app) = train_test_split(Data_txt['text'],Data_txt['label'],test_size=0.4)
(X_test,X_val,y_test,y_val) = train_test_split(X_app,y_app,test_size=0.5)



############### Bow##############################

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_vecteur = vectorizer.transform(X_train)
X_val_vecteur = vectorizer.transform(X_val)
X_test_vecteur= vectorizer.transform(X_test)


############### TFIDF #######################


tf_transformer = TfidfTransformer().fit(X_train_vecteur)

X_train_tf = tf_transformer.transform(X_train_vecteur)
X_val_tf = tf_transformer.transform(X_val_vecteur)
X_test_tf = tf_transformer.transform(X_test_vecteur)




#################### Naive bayes (bof) #######################

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


Multi_nb = MultinomialNB()
Multi_nb.fit(X_train_vecteur,y_train)

y_pred_train = Multi_nb.predict(X_train_vecteur)
print("Precision(Phase de train) : ",accuracy_score(y_train,y_pred_train))

y_pred_val = Multi_nb.predict(X_val_vecteur)
print("Precision(Phase de validation) : ",accuracy_score(y_val,y_pred_val))

y_pred_test = Multi_nb.predict(X_test_vecteur)
print("Precision (Phase de test) : ",accuracy_score(y_test,y_pred_test),"\n")


print(classification_report(y_test,y_pred_test))
print(confusion_matrix(y_test,y_pred_test))




########## Naive bayes ##########################


Multi_nb.fit(X_train_tf, y_train)

pred_train_tf = Multi_nb.predict(X_train_tf)
pred_val_tf = Multi_nb.predict(X_val_tf)
pred_test_tf = Multi_nb.predict(X_test_tf)



print("Precision(Phase de train) : : ", accuracy_score(y_train, pred_train_tf))

print("Precision(Phase de validation)t : ", accuracy_score(y_val, pred_val_tf))

print("Precision(Phase de test) : ", accuracy_score(y_test, pred_test_tf))


print(classification_report(y_test,pred_test_tf))
print(confusion_matrix(y_test,pred_test_tf))

########### Mlp Classifier ########################
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(activation='relu', alpha=1.0, verbose=2, batch_size=50)

MLP.fit(X_train_vecteur, y_train)

pred_tmlp = MLP.predict(X_train_vecteur)
pred_vmlp = MLP.predict(X_val_vecteur)
pred_temlp = MLP.predict(X_test_vecteur)



print("Precision(Phase de train) : ", accuracy_score(y_train, pred_tmlp))

print("Precision(Phase de validation) : ", accuracy_score(y_val, pred_vmlp))

print("Precision(Phase de test): ", accuracy_score(y_test, pred_temlp))

