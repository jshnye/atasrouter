from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import jaccard_score
from scipy.sparse.linalg import norm
import os, openai
import numpy as np


from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

class fixSS:
    def __init__(self):
        self.query_text: str
        self.openai_api_key = openai.api_key
        self.openai_organization = openai.organization 
        
    def Normalize_jaccard(self):

        tfidf_list = []
        dir = "ATAS/prompt_in_text"
        # only load .txt file
        txt_files = [f for f in os.listdir(dir) if f.endswith(".txt")]

        for file in txt_files:
            print("Reading File: ", file)
            try:
                with open(os.path.join(dir, file), 'r') as file:
                    documents = file.read()
                    # print(documents)        
                    tfidf_list.append(documents)
                    
            except FileNotFoundError:
                print(f"The file '{file}' was not found.")
            except Exception as e:
                print(f"An error occurred while processing '{file}': {str(e)}")
        
        # print(tfidf_list)
        
        # Create TF-IDF vectors
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_list).toarray()
        print(type(tfidf_matrix))
        print(np.shape(tfidf_matrix))
        print(tfidf_matrix)
        
        
        
        embedding_file_path = "ATAS/return_and_refund.npy"
        embeded_doc = np.load(embedding_file_path)
        print(type(embeded_doc))
        print(np.shape(embeded_doc))

        # # Normalize document vectors
        # # Normalize document vectors
        # normalized_tfidf_matrix = tfidf_matrix / norm(tfidf_matrix, axis=1)

        # # Compute cosine similarity
        # cosine_similarity_score = cosine_similarity(normalized_tfidf_matrix, embedded_query)
        # jaccard_similarity_score = jaccard_score(normalized_tfidf_matrix, embedded_query)
        # print("Cosine: ",cosine_similarity_score)
        # print("jaccard: ", jaccard_similarity_score)
                              
                
if __name__ == "__main__":
    fix = fixSS()
    fix.Normalize_jaccard()



