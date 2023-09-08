import pandas as pd
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

stop = stopwords.words('english')


movieReviewTable = pd.read_csv('IMDB dataset.csv')
movieReviewTable.head()
movieReviewTable['review'] = movieReviewTable['review'].str.lower()
movieReviewTable['review'] = movieReviewTable['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


smsSpamTable = pd.read_csv('sms_spam.csv')
smsSpamTable.head()
smsSpamTable['text'] = smsSpamTable['text'].str.lower()
smsSpamTable['text'] = smsSpamTable['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


emailSpamTable = pd.read_csv('spam_ham_dataset.csv')
emailSpamTable.head()
emailSpamTable['text'] = emailSpamTable['text'].str.lower()
emailSpamTable['text'] = emailSpamTable['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

twitterTable = pd.read_csv('twitter_training.csv')
twitterTable.head()
twitterTable['text'] = twitterTable['text'].str.lower()
twitterTable['text']= twitterTable['text'].fillna("")
twitterTable['text'] = twitterTable['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

yelpTable = pd.read_csv('test.csv')
twitterTable.head()
yelpTable['text'] = yelpTable['text'].str.lower()
yelpTable['text'] = yelpTable['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


from sklearn.preprocessing import LabelEncoder
movieReviewTable['sentiment'] = movieReviewTable['sentiment'].astype('category')
type_encode = LabelEncoder()
movieReviewTable['sentiment'] = type_encode.fit_transform(movieReviewTable.sentiment)

smsSpamTable['type'] = smsSpamTable['type'].astype('category')
type_encode2 = LabelEncoder()
smsSpamTable['type'] = type_encode2.fit_transform(smsSpamTable.type)

twitterTable['sentiment'] = twitterTable['sentiment'].astype('category')
type_encode3 = LabelEncoder()
twitterTable['sentiment'] = type_encode3.fit_transform(twitterTable.sentiment)





from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer()
col1 = movieReviewTable["review"]
col1array = vectorizer1.fit_transform(col1)
col2 = movieReviewTable["sentiment"]

vectorizer2 = CountVectorizer()
col3 = smsSpamTable["text"]
col3array = vectorizer2.fit_transform(col3)
col4 = smsSpamTable["type"]

vectorizer3 = CountVectorizer()
col5 = emailSpamTable["text"]
col5array = vectorizer3.fit_transform(col5)
col6 = emailSpamTable["label_num"]

vectorizer4 = CountVectorizer()
col7 = twitterTable["text"]
col7array = vectorizer4.fit_transform(col7.astype('U'))
col8 = twitterTable["sentiment"]

vectorizer5 = CountVectorizer()
col9 = yelpTable["text"]
col9array = vectorizer5.fit_transform(col9)
col10 = yelpTable["sentiment"]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Ytest = train_test_split(col1array, col2, test_size=.2, random_state=0)

x2Train, x2Test, y2Train, y2Test = train_test_split(col3array, col4, test_size=.2, random_state=0)

x3Train, x3Test, y3Train, y3Test = train_test_split(col5array, col6, test_size=.2, random_state=0)

x4Train, x4Test, y4Train, y4Test = train_test_split(col7array, col8, test_size=.2, random_state=0)

x5Train, x5Test, y5Train, y5Test = train_test_split(col9array, col10, test_size=.2, random_state=0)


movieReviewModel = LogisticRegression(solver='sag', penalty='l2')
movieReviewModel.fit(X_train, Y_train)
pred = movieReviewModel.predict(X_test)

smsSpamModel = LogisticRegression(solver='sag', penalty='l2')
smsSpamModel.fit(x2Train, y2Train)
pred2 = smsSpamModel.predict(x2Test)

emailSpamModel = LogisticRegression(solver='liblinear', penalty='l1')
emailSpamModel.fit(x3Train, y3Train)
pred3 = emailSpamModel.predict(x3Test)

twitterModel = LogisticRegression(solver='liblinear', penalty='l1')
twitterModel.fit(x4Train, y4Train)
pred4 = twitterModel.predict(x4Test)

yelpModel = LogisticRegression(solver='sag', penalty='l2')
yelpModel.fit(x5Train, y5Train)
pred5 = yelpModel.predict(x5Test)


from sklearn.metrics import accuracy_score
print('\nMovie Review dataset accuracy: {:.1f}%'.format(accuracy_score(Ytest, pred)*100))

print('\nSMS Spam dataset accuracy: {:.1f}%'.format(accuracy_score(y2Test, pred2)*100))

print('\nEmail Spam dataset accuracy: {:.1f}%'.format(accuracy_score(y3Test, pred3)*100))

print('\nTwitter Sentiment dataset accuracy: {:.1f}%'.format(accuracy_score(y4Test, pred4)*100))

print('\nYelp Sentiment dataset accuracy: {:.1f}%'.format(accuracy_score(y5Test, pred5)*100))

