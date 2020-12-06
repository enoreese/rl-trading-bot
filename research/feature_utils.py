from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import multiprocessing
from numpy import hstack
from tqdm import notebook
import pandas as pd
from pymagnitude import *
from scipy import sparse
from nltk.corpus import stopwords


def readability_scores_mp(data):
    result_dict, idx, text = data

    #  flesch_reading_ease =  textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)

    result_dict[idx] = [flesch_kincaid_grade, dale_chall_readability_score]


def calc_readability_scores(df):
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    mp_list = [(result_dict, idx, text) for idx, text in enumerate(df.clean_tweet.values)]

    with multiprocessing.Pool(os.cpu_count()) as p:
        r = list(notebook.tqdm(p.imap(readability_scores_mp, mp_list), total=len(mp_list)))
    rows = [result_dict[idx] for idx in range(df.clean_tweet.values.shape[0])]
    return pd.DataFrame(rows).values


def text_features(df):
    longest_word_length = []
    mean_word_length = []
    length_in_chars = []

    for text in df.clean_tweet.values:
        length_in_chars.append(len(text))
        longest_word_length.append(len(max(text.split(), key=len)))
        mean_word_length.append(np.mean([len(word) for word in text.split()]))

    longest_word_length = np.array(longest_word_length).reshape(-1, 1)
    mean_word_length = np.array(mean_word_length).reshape(-1, 1)
    length_in_chars = np.array(length_in_chars).reshape(-1, 1)

    return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis=1)


def word_ratio(df):
    with open('data/DaleChallEasyWordList.txt') as f:
        easy_words_list = [line.rstrip(' \n') for line in f]

    with open('data/terrier-stopword.txt') as f:
        terrier_stopword_list = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += stopwords.words('english')

    with open('data/common.txt') as f:
        common = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += common

    easy_words_ratio = []
    stop_words_ratio = []

    for text in df.clean_tweet.values:
        easy_words = 0
        stop_words = 0
        total_words = 0

        for word in text.split():
            if word.lower() in easy_words_list:
                easy_words += 1
            if word.lower() in terrier_stopword_list:
                stop_words += 1
            total_words += 1

        easy_words_ratio.append(easy_words / total_words)
        stop_words_ratio.append(stop_words / total_words)

    easy_words_ratio = np.array(easy_words_ratio).reshape(-1, 1)
    stop_words_ratio = np.array(stop_words_ratio).reshape(-1, 1)

    return np.concatenate(
        [easy_words_ratio, stop_words_ratio], axis=1)


def calc_sentiment_scores(df):
    sid = SentimentIntensityAnalyzer()
    neg = []
    neu = []
    pos = []
    compound = []

    for text in df.clean_tweet.values:
        sentiments = sid.polarity_scores(text)
        neg.append(sentiments['neg'])
        neu.append(sentiments['neu'])
        pos.append(sentiments['pos'])
        compound.append(sentiments['compound'])

    neg = np.array(neg).reshape(-1, 1)
    neu = np.array(neu).reshape(-1, 1)
    pos = np.array(pos).reshape(-1, 1)
    compound = np.array(compound).reshape(-1, 1)
    return np.concatenate([neg, pos, compound], axis=1)


def get_glove_vectors(df, glove):
    vectors = []
    for text in notebook.tqdm(df.clean_tweet.values):
        vectors.append(np.average(glove.query(word_tokenize(text)), axis=0))
    return np.array(vectors)


def tfidf_w2v(df, idf_dict, glove):
    vectors = []
    for title in notebook.tqdm(df.clean_tweet.values):
        w2v_vectors = glove.query(word_tokenize(title))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(title)]
        vectors.append(np.average(w2v_vectors, axis=0, weights=weights))
    return np.array(vectors)


def featurize_text(df, embedding_type='glove'):
    df_text_features = text_features(df)
    df_word_ratio = word_ratio(df)
    df_sentiment_scores = calc_sentiment_scores(df)
    df_readability = calc_readability_scores(df)

    feature_names = list(df.columns)

    embedder = None
    if embedding_type == 'tfidf':
        print('TFIDF Title....')

        embedder = TfidfVectorizer()

        print('TFIDF Word....')
        embedding_features = embedder.fit_transform(df.text.values)

    elif embedding_type == 'bow':
        print('BOW Title....')

        embedder = CountVectorizer()

        print('BOW Word....')
        embedding_features = embedder.fit_transform(df.text.values)

    elif embedding_type == 'glove':
        print('Glove.....')
        glove = Magnitude("/Users/osasusen/Dev/RLBot/research/vectors/glove.twitter.27B.100d.magnitude")
        data_glove = get_glove_vectors(df, glove)
        print(data_glove.shape)
        embedding_features = sparse.csr_matrix(data_glove)

    elif embedding_type == 'tfidf_glove':
        print('Glove.....')

        glove = Magnitude("/Users/osasusen/Dev/RLBot/research/vectors/glove.twitter.27B.100d.magnitude")

        tfidf = TfidfVectorizer()
        tfidf.fit(df.clean_tweet.values)
        idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

        data_glove = tfidf_w2v(df, idf_dict, glove)

        # embedding_features = sparse.csr_matrix(data_glove)

    feature_names.extend([
        'longest_word_length',
        'mean_word_length',
        'length_in_chars',
        'easy_words_ratio',
        'stop_words_ratio',
        'sentiment_neg',
        'senitment_pos',
        'sentiment_compound',
        'flesch_kincaid_grade',
        'dale_chall_readability_score'
    ])

    featurs = hstack((
        df.values,
        df_text_features,
        df_word_ratio,
        df_sentiment_scores,
        df_readability
    ))
    print(featurs[:2])

    print(embedding_features[:2])
    print(data_glove.shape)

    # features = sparse.csr_matrix(featurs)
    # features = sparse.hstack((
    #     featurs,
    #     embedding_features
    # ))
    # print(features)

    if embedding_type == 'tfidf':
        embedd_names = ['tfidf_word_' + col for col in embedder.get_feature_names()]
    elif embedding_type == "glove":
        embedd_names = ['glove_' + str(col) for col in range(100)]
    elif embedding_type == "bow":
        embedd_names = ['bow_word_' + col for col in embedder.get_feature_names()]
    print('DONE!')

    print(len(embedd_names))

    new_features = pd.DataFrame(featurs, columns=feature_names)
    embedd_df = pd.DataFrame(data_glove, columns=embedd_names)

    new_df = pd.concat([new_features, embedd_df], axis=1)
    return new_df
