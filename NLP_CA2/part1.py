import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('tweets_data.csv', header=None, usecols=[0, 5], names=['target', 'text'], encoding='latin1')
df.dropna(subset=['text'], inplace=True)
df['text'] = df['text'].apply(lambda x: x.strip()).astype(str)
df = df[df['text'] != '']

negative_sample = df[df['target'] == 0].sample(n=5000, random_state=1)
positive_sample = df[df['target'] == 4].sample(n=5000, random_state=1)
sampled_tweets = pd.concat([negative_sample, positive_sample]).sample(frac=1, random_state=1).reset_index(drop=True)

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and not token.is_punct and not token.like_url])

sampled_tweets['text'] = sampled_tweets['text'].apply(preprocess_text)

train_data, test_data = train_test_split(sampled_tweets, test_size=0.2, random_state=1)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])

feature_names = vectorizer.get_feature_names_out()
dense_matrix = X_train.toarray()
df_count_vectorized = pd.DataFrame(dense_matrix, columns=feature_names)
csv_filename = 'TF_matrix.csv'
df_count_vectorized.to_csv(csv_filename, index=True)

clf = MultinomialNB().fit(X_train, train_data['target'].astype('int'))
yhat = clf.predict(vectorizer.transform(test_data['text']))

print("***TF Model Metrics:***")
print(f"Accuracy: {accuracy_score(test_data['target'].astype('int'), yhat)}")
print(f"Precision: {precision_score(test_data['target'].astype('int'), yhat, pos_label=4)}")
print(f"Recall: {recall_score(test_data['target'].astype('int'), yhat, pos_label=4)}")
print(f"F1-score: {f1_score(test_data['target'].astype('int'), yhat, pos_label=4)}\n")

def compute_tfidf(X_counts):
    counts_array = X_counts.toarray()
    doc_sum = X_counts.sum(axis=1).A1
    eps = 1e-6
    tf = counts_array / np.where(doc_sum[:, None] == 0, eps, doc_sum[:, None])
    df = np.sum(counts_array > 0, axis=0)
    num_docs = X_counts.shape[0]
    idf = np.log((num_docs + 1) / (df + 1)) + 1
    idf_array = np.array(idf).flatten()
    tfidf = tf * idf_array
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + eps
    normalized_tfidf = tfidf / norms
    return np.nan_to_num(normalized_tfidf)

X_train_tfidf_manual = compute_tfidf(X_train)

df_tfidf = pd.DataFrame(X_train_tfidf_manual, columns=feature_names)
tfidf_csv_filename = 'TFIDF_matrix.csv'
df_tfidf.to_csv(tfidf_csv_filename, index=True)

X_test_tfidf_manual = compute_tfidf(vectorizer.transform(test_data['text']))
clf_tfidf_manual = MultinomialNB().fit(X_train_tfidf_manual, train_data['target'].astype('int'))
yhat_tfidf_manual = clf_tfidf_manual.predict(X_test_tfidf_manual)

print("***TF-IDF Model Metrics:***")
print(f"Accuracy: {accuracy_score(test_data['target'].astype('int'), yhat_tfidf_manual)}")
print(f"Precision: {precision_score(test_data['target'].astype('int'), yhat_tfidf_manual, pos_label=4)}")
print(f"Recall: {recall_score(test_data['target'].astype('int'), yhat_tfidf_manual, pos_label=4)}")
print(f"F1-score: {f1_score(test_data['target'].astype('int'), yhat_tfidf_manual, pos_label=4)}\n")


def compute_ppmi_matrix(X_counts):
    eps = 1e-6
    total_count = np.sum(X_counts)
    sum_over_words = np.array(X_counts.sum(axis=0)).flatten()
    sum_over_docs = np.array(X_counts.sum(axis=1)).flatten()

    expected = np.outer(sum_over_docs, sum_over_words) / total_count
    X_counts = X_counts.toarray()
    pmi_matrix = np.maximum(0, np.log2((X_counts * total_count) / (expected + eps) + eps))
    ppmi_matrix = np.maximum(0, pmi_matrix)

    return ppmi_matrix

X_train_ppmi = compute_ppmi_matrix(X_train)

df_ppmi = pd.DataFrame(X_train_ppmi, columns=feature_names)
ppmi_csv_filename = 'PPMI_matrix.csv'
df_ppmi.to_csv(ppmi_csv_filename, index=True)

X_test_ppmi = compute_ppmi_matrix(vectorizer.transform(test_data['text']))
clf_ppmi = MultinomialNB().fit(X_train_ppmi, train_data['target'].astype('int'))
yhat_ppmi = clf_ppmi.predict(X_test_ppmi)

print("***PPMI Model Metrics:***")
print(f"Accuracy: {accuracy_score(test_data['target'].astype('int'), yhat_ppmi)}")
print(f"Precision: {precision_score(test_data['target'].astype('int'), yhat_ppmi, pos_label=4)}")
print(f"Recall: {recall_score(test_data['target'].astype('int'), yhat_ppmi, pos_label=4)}")
print(f"F1-score: {f1_score(test_data['target'].astype('int'), yhat_ppmi, pos_label=4)}")