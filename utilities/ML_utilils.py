from sklearn.feature_extraction.text import CountVectorizer

def featurize_bow(X, binary=True):
    vec = CountVectorizer(binary=binary)
    return vec.fit_transform(X)
