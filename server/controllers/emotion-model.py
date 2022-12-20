import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import sys
import joblib


df = pd.read_csv("emotion-dataset.csv")
df.head()

df['Emotion'].value_counts()

dir(nfx)

print(df)


df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(Xfeatures,ylabels, test_size=0.2, random_state=42)

pipe_lr.fit(X_train,y_train)




filename = 'finalized_model.sav'
joblib.dump(pipe_lr, filename)


# val=sys.argv[1]

# print(pipe_lr.predict([val]))