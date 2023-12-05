from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

def featurize_bow(X, binary=True):
    vec = CountVectorizer(binary=binary)
    return vec.fit_transform(X)

def featurize_tfIDF(X_train, max_pct_freq, min_count_freq):
    vec = TfidfVectorizer(max_df=max_pct_freq, min_df=min_count_freq)
    return vec.fit_transform(X_train)