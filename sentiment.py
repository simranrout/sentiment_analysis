import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
import pandas as pd
#nltk.download('stopwords')

dataset= pd.read_csv('Dataset_final.csv')
x=dataset.User
y=dataset.U_Sentiment


corpus=[]
for i in range(0,len(x)):
    temp=re.sub(r'\W',' ',str(x[i]))
    temp= temp.lower()
    if len(temp)>=1 and(temp.isdigit() ):
        x.drop([i],axis=0,inplace=True )
    else:
        temp =re.sub(r'\s+[a-z]\s+',' ',temp)
        temp=re.sub(r'^[a-z]\s+',' ',temp)
        temp=re.sub(r'\s+',' ',temp)
        corpus.append(temp)
        

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
vectorizer= CountVectorizer(max_features=500, min_df=3, max_df=0.4,stop_words=stopwords.words('english') )
x=vectorizer.fit_transform(corpus).toarray()


transformer= TfidfTransformer()
x=transformer.fit_transform(x).toarray()



from sklearn.preprocessing import LabelEncoder
#here we encode all the text value to numerical value to encode it we have used LabelEncoder
labelencoder_x1=LabelEncoder()
y=labelencoder_x1.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 , random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier =MultinomialNB()
classifier.fit(x_train,y_train)


y_pred= classifier.predict(x_test)


from sklearn.metrics import confusion_matrix,classification_report

cm= confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))














