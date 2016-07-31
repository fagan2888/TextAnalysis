import re
import string

import nltk

from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def get_words(text):
    """
    Tokenizes the text string, and returns a list of tokens with
    one or more alphanumeric characters.

    Parameters
    ----------
    text: A string.

    Returns
    -------
    words: A list of strings.
    """

    pattern = re.compile(r'[^\w\s]')
    words = re.sub(pattern, ' ', text.lower()).split()

    return words


def count(words):
    """
    Computes the the number of token, number of words, and lexical diversity.

    Parameters
    ----------
    words: A list of of strings.

    Returns
    -------
    A 3-tuple of (num_tokens, num_words, lex_div)
    num_tokens: An int. The number of tokens in "words".
    num_words: An int. The number of words in "words".
    lex_div: A float. The lexical diversity of "words".
    """

    counts = nltk.FreqDist(words)
    num_words = len(words)
    num_tokens = len(counts)
    lex_div = num_words / num_tokens

    return num_tokens, num_words, lex_div


def get_most_common(words, ntop):
    """
    Computes the most commonly occurring terms and their counts.

    Parameters
    ----------
    words: A list of of strings.
    ntop: An int. The number of most common words that will be returned.

    Returns
    -------
    A list of tuple (token, frequency).
    """

    counts = nltk.FreqDist(words)
    most_common = counts.most_common(ntop)

    return most_common


def find_hapaxes(words):
    """
    Finds hapexes in "words".

    Parameters
    ----------
    words: A list of strings.

    Returns
    -------
    A list of strings.
    """

    counts = nltk.FreqDist(words)
    hapaxes = counts.hapaxes()

    return hapaxes


def count_corpus(corpus):
    """
    Computes the the number of token, number of words, and lexical diversity.

    Parameters
    ----------
    corpus: An NLTK corpus.

    Returns
    -------
    A 3-tuple of (num_tokens, num_words, lex_div)
    num_tokens: An int. The number of tokens in "words".
    num_words: An int. The number of words in "words".
    lex_div: A float. The lexical diversity of "words".
    """

    num_tokens, num_words, lex_div = count(corpus.words())

    return num_words, num_tokens, lex_div


def get_long_words(corpus, length=20):
    """
    Finds all words in "corpus" longer than "length".

    Parameters
    ----------
    corpus: An NLTK corpus.
    length: An int. Default: 22

    Returns
    -------
    A list of strings.
    """

    return [word for word in corpus.words() if len(word) > length]


def get_categories(corpus):
    """
    Finds categories of an NLTK corpus.

    Parameters
    ----------
    corpus: An NLTK corpus object.

    Returns
    -------
    A list of strings.
    """

    categories = corpus.categories()

    return categories


def get_fileids(corpus):
    """
    Finds all fileids of an NLTK corpus.

    Parameters
    ----------
    corpus: An NLTK corpus object.

    Returns
    -------
    A list of strings.
    """

    fileids = corpus.fileids()

    return fileids


def get_categories_from_fileids(corpus, fileids):
    """
    Finds categories for each element of "fileids".

    Parameters
    ----------
    corpus: An NLTK corpus.
    fileids: A list of strings.

    Returns
    -------
    A list of strings.
    """

    result = list()

    for fileid in fileids:
        result.append(corpus.categories(fileids=fileid)[0])

    return result


def train_test_split(corpus, fileids, categories):
    """
    Creates a training set and a test from the NLTK Reuters corpus.

    Parameters
    ----------
    corpus: An NLTK corpus.
    fileids: A list of strings.
    categories: A list of strings.

    Returns
    -------
    A 4-tuple (X_train, X_test, y_train, y_test)
    All four elements in the tuple are lists of strings.
    """

    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()

    for fileid in fileids:
        if fileid.startswith('train'):
            X_train.append(corpus.raw(fileids=fileid))
            y_train.append(corpus.categories(fileids=fileid)[0])
        else:
            X_test.append(corpus.raw(fileids=fileid))
            y_test.append(corpus.categories(fileids=fileid)[0])

    return X_train, X_test, y_train, y_test


def cv_svc(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (cv, sv, y_pred)
    cv: A CountVectorizer instance.
    svc: A LinearSVC instance.
    y_pred: A numpy array.
    """

    cv = CountVectorizer()

    train_counts = cv.fit_transform(X_train)
    test_data = cv.transform(X_test)

    svc = LinearSVC(random_state=random_state)

    clf = svc.fit(train_counts, y_train)
    y_pred = clf.predict(test_data)

    return cv, svc, y_pred


def cv_svc_pipe(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    """

    tools = [('cv', CountVectorizer()), ('svc', LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    return clf, predicted


def cv_svc_pipe_sw(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.
    Uses English stop words.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    """

    tools = [('cv', CountVectorizer(stop_words='english')), ('svc', LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    return clf, predicted


def tfidf_svc(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.
    Uses English stop words.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    """

    tools = [('tf', TfidfVectorizer(stop_words='english')), ('svc', LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)

    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    return clf, predicted


def ngram(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.
    Uses unigrams, bigrams, and trigrams.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    """

    tools = [('tf', TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=.75, min_df=3)),
             ('svc', LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, y_pred


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    stemmer = PorterStemmer()
    stems = map(stemmer.stem, tokens)

    return stems


def stem(X_train, y_train, X_test, random_state):
    """
    Creates a document term matrix and uses SVM classifier to make document classifications.
    Uses the Porter stemmer.

    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    """

    tools = [('tf', TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_df=.5, min_df=2, tokenizer=tokenize)),
             ('svc', LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, y_pred


def cluster(X_train, X_test, true_k, random_state):
    """
    Applies clustering analysis to a feature matrix.

    Parameters
    ----------
    X_train: A list of strings.
    X_test: A list of strings.
    true_k: An int. The number of clusters.
    random_state: A np.random.RandomState instance.

    Returns
    -------
    A Pipeline instance.
    """
    tools = [('tf', TfidfVectorizer(ngram_range=(1, 1), stop_words='english')),
             ('km', KMeans(n_clusters=true_k, random_state=random_state))]
    clf = Pipeline(tools)
    clf = clf.fit(X_train)

    return clf


def get_top_tokens(km, tf, cluster, top_tokens):
    """
    Identifies the most frequently used words in "cluster".

    Parameters
    ----------
    km: A Kmeans instance.
    tf: A TfidfVectorizer instance.
    icluster: An int. Which cluster?
    top_tokens: An int. How many tokens do you want?

    Returns
    -------
    A list of strings.
    :param cluster:
    """

    tokens = list()

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tf.get_feature_names()

    for token in order_centroids[cluster, :top_tokens]:
        tokens.append(terms[token])

    return tokens
