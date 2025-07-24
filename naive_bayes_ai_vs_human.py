import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv("data.csv")

x=df["text"]
y=df["label"]

vectorizer=CountVectorizer()
x_vec=vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_vec,y,test_size=0.2,random_state=42)

model=MultinomialNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("accuracy rate:",accuracy_score(y_test,y_pred))
print("\ndetailed report\n",classification_report(y_test,y_pred))

def tahmin_et(metin):
    metin_vec=vectorizer.transform([metin])
    sonuc=model.predict(metin_vec)[0]
    print(sonuc)

tahmin_et("Write any sentence you want here and try it")
