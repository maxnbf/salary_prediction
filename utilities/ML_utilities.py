from sklearn.feature_extraction.text import CountVectorizer

def featurize_bow(X):
    vec = CountVectorizer(binary=True)
    return vec.fit_transform(X)
