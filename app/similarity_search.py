import json, os, openai, time, asyncio, sys
import numpy as np  # Assuming the embedding is a NumPy array
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity
from sklearn.metrics import jaccard_score
from langchain.embeddings import OpenAIEmbeddings
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# from DATA_Chatbot.Chatbot_DATA_QA import Chatbot_DATA_QA
# from DOC_Chatbot.Chatbot_DOC_QA import Chatbot_DOC_QA


from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")



class SimilaritySearch:

    '''
    This class is to do similarity search using an already convert-to-embedding context
    
    
    '''
    
    def __init__(self, 
                #  query: str, 
                #  chat_history: list,
                #  language: str,
                 brand_name: str,
                 ):
        self.npfile_parent_dir = "TEMP"
        self.sub_dir = brand_name
        self.openai_api_key = openai.api_key
        self.openai_organization = openai.organization 
        self.array_length: int
        # self.query_text = query
        # self.chat_history = chat_history
        # self.language = language
    
    
    def sum_of_emb_length(self):
        dir = 'app/ATAS'
        sum_emb_length = 0
        for filename in os.listdir(dir):
            file = np.load(os.path.join(dir, filename))
            sum_emb_length += np.shape(file)[0]
        # print(sum_emb_length)
        return sum_emb_length    
    
    
    def similarity_search_by_embedding(self):
        dir = 'app/ATAS'
        atas_similarity_score = []
        atas_filename = []    
        
        for filename in os.listdir(dir):
            print("Compare Document: ",filename)
            atas_filename.append(filename)
            embeded_doc = np.load(os.path.join(dir, filename))
            self.array_length = ((np.shape(embeded_doc))[0])
            print("array size: ",self.array_length)
            
            embeddings_model = OpenAIEmbeddings()
            self.query_text = 'can I make payment using touch n go?'
            embedded_query = embeddings_model.embed_documents(self.query_text)
            embedded_query = np.array(embedded_query)

            similarity_score = np.mean(cosine_similarity(embedded_query, embeded_doc))
            # percentage = self.array_length/self.sum_of_emb_length()
            # normalize_similarity_score = similarity_score * percentage
            print(f'normalize score:  {similarity_score} ')
            atas_similarity_score.append(similarity_score)
            print()
          
        highest_similarity_acore = max(atas_similarity_score)    
        for index, element in enumerate(atas_similarity_score):
            if element == highest_similarity_acore:
                print('Routing to ', atas_filename[index])
               

            
                
   
if __name__ == "__main__":
    search = SimilaritySearch(brand_name='ATAS')
    search.similarity_search_by_embedding()
    # search.sum_of_emb_length()
  



