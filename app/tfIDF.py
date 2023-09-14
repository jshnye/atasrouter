import string, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def count_unique_words(text):
    # Remove punctuation marks from the text
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split the text into words using whitespace as a delimiter
    words = text.split()

    # Use a set to store unique words
    unique_words = set(words)

    # Return the count of unique words
    print(len(unique_words))
    print(unique_words)
    return len(unique_words)


def count_doc_word():
    dir = "ATAS/prompt_in_text/return_and_refund.txt"
    with open (dir, 'r') as file :
        doc = file.read()
        
        count_unique_words(doc)


def tf_idf():
    corpus = []
    dir = "ATAS/prompt_in_text"
    txt_files = [f for f in os.listdir(dir) if f.endswith(".txt")]
    for file in txt_files:
        with open (os.path.join(dir,file), 'r') as file :
            doc = file.read()
            corpus.append(doc)
            count_unique_words(doc)        
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    x = []
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    for keys, values in tfidf_df.items():
        x.append(values[2]) 
    print(len(x))


def load_common_embedding():
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_list).toarray()
    # print(type(tfidf_matrix))
    # print(np.shape(tfidf_matrix))
    # print(tfidf_matrix)
    dir = "ATAS"
    # only load .txt file
    npy_files = [f for f in os.listdir(dir) if f.endswith(".npy")]
    for file in npy_files: 
        print("File name: ",file)
        embeded_doc = np.load(os.path.join(dir,file))
        print(type(embeded_doc))
        print(np.shape(embeded_doc))


# def w2v_embedding():
    # model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)




# Example usage:
# tf_idf()
load_common_embedding()