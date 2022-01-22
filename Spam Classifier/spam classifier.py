
import pandas as pd

#importing the dataset
messages = pd.read_csv('E:/jupyterfiles/Practice/NLP/Spam Classifier/smsspamcollection/SMSSpamCollection',sep='\t',
                       names=["labels","message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#preprocessing messages are storing in corpus variable
corpus = []
for i in range(0,len(messages)):
    
    #removing unnecessary character except alphabets and replace with space
    final = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    
    #lowering the text
    final = final.lower()
    
    #splitting the text into each word for stemming
    final = final.split()
    
    #applying stemming
    final = [ps.stem(word) for word in final if not word in stopwords.words('english')]
    
    #joining the stemming words into sentence
    final  = ' '.join(final)
    
    #appending to corpus variable
    corpus.append(final)
    
#creating Bag of words
from sklearn.feature_extraction.text import CountVectorizer

#selecting top 5000 words in columns 
cv=CountVectorizer(max_features=5000)

X = cv.fit_transform(corpus).toarray()

#converting categorical columns using label encoding
y = pd.get_dummies(messages['labels'])
y = y.iloc[:,1].values

#train & test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# training model using NavieBayes classifier
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
spam_classifier_model = nb.fit(X_train,y_train)

#testing the trained model
prediction = spam_classifier_model.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix( y_test, prediction)

#checking the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)












