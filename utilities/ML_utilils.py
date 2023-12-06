from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

def featurize_bow(X, binary=True):
    """Featurizes a list of a list of strings by using a bag-of-words methodology.

    Args:
        X (2D matrix (list of list)): list of a list of strings to be featurized
        binary (bool, optional): Whether to use binary BOW or count BOW. Defaults to True.

    Returns:
        2D matrix (list of list): A featurized representation of the strings passed in.
    """
    vec = CountVectorizer(binary=binary)
    return vec.fit_transform(X)

def featurize_tfIDF(X_train, max_pct_freq, min_count_freq):
    """Featurizes a list of a list of strings by using a TfIDF methodology.

    Args:
        max_pct_freq (float, <=1.0): the maximum pct of documents a word should appear in to be included
        min_count_freq (int): the minimum count a word should appear in a single document to be included
        X (2D matrix (list of list)): list of a list of strings to be featurized

    Returns:
        2D matrix (list of list): A featurized representation of the strings passed in.
    """
    vec = TfidfVectorizer(max_df=max_pct_freq, min_df=min_count_freq)
    return vec.fit_transform(X_train)