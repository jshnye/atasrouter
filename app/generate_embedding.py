import openai, os, asyncio, time
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class GenerateEmbeddingFile:
    '''
    This class is to generate embedding for the prompt that are orignally stored in txt/str format.
    
    Objective: 
    reduce time needed to process text to embedding, 
    especially context is long  
    map reuce is applied to overcome token limit problem
    '''
    
    
    def __init__(self, brand_name:str):
        self.txtfile_parent_dir = "router/text_store"
        self.sub_dir = brand_name
        self.npfile_parent_dir = "router/embedding_store"


    def text_to_embedding(self, txt_filename, embedding_filename):
        file = os.path.join(self.txtfile_parent_dir, self.sub_dir , txt_filename)
        print()
        print("Here!!!")
        print(file)
        raw_documents = TextLoader(file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
        documents = text_splitter.split_documents(raw_documents)
        # print(documents[0].page_content)

        # embed the document and store the embedding 
        embeddings_model = OpenAIEmbeddings()

        embeded_doc = embeddings_model.embed_documents(documents[0].page_content)
        embedded_doc = np.array(embeded_doc)
        print(type(embedded_doc))

        embedding_file= os.path.join(self.npfile_parent_dir, self.sub_dir, embedding_filename)
 
        # Save the embedding data to a numpy file
        with open(embedding_file, "wb") as file:
            np.save(file, embedded_doc)

    async def embed_prompt(self, txt_filename, embedding_filename):
        self.text_to_embedding(txt_filename=txt_filename, embedding_filename=embedding_filename)
    
    # async def embed_checkout_assistant_prompt(self):
    #     txt_filename = "checkout_assistant.txt"
    #     embedding_filename = "checkout_assistant.npy"
    #     self.text_to_embedding(txt_filename=txt_filename, embedding_filename=embedding_filename)
        
    # async def embed_gmail_prompt(self):
    #     txt_filename = "gmail.txt"
    #     embedding_filename = "gmail.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)

    # async def embed_google_calendar_prompt(self):
    #     txt_filename = "google_calendar.txt"
    #     embedding_filename = "google_calendar.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)
        
    # async def embed_order_tracking_prompt(self):
    #     txt_filename = "order_tracking.txt"
    #     embedding_filename = "order_tracking.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)
        
    # async def embed_return_and_refund_prompt(self):
    #     txt_filename = "return_and_refund.txt"
    #     embedding_filename = "return_and_refund.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)
        
    # async def embed_shopping_cart_prompt(self):
    #     txt_filename = "shopping_cart.txt"
    #     embedding_filename = "shopping_cart.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)
        
    # async def embed_user_profile_management_prompt(self):
    #     txt_filename = "user_profile_management.txt"
    #     embedding_filename = "user_profile_management.npy"
    #     self.text_to_embedding(txt_filename, embedding_filename)
        

    
    def run(self):
        start = time.time()

        # Create an event loop
        loop = asyncio.get_event_loop()
        # Run the functions asynchronously
        text_filenames = [
            "checkout_assistant.txt",
            "gmail.txt",
            "google_calendar.txt",
            "order_tracking.txt",
            "return_and_refund.txt",
            "shopping_cart.txt",
            'user_profile_management.txt'
        ]
        
        npy_filenames = [
            "checkout_assistant.npy",
            "gmail.npy",
            "google_calendar.npy",
            "order_tracking.npy",
            "return_and_refund.npy",
            "shopping_cart.npy",
            'user_profile_management.npy'
        ]
        
        tasks = [
            self.embed_prompt(
                txt_filename=txt_filename,
                embedding_filename=npy_filename
            )
            for txt_filename, npy_filename in zip(
                text_filenames, npy_filenames
            )
        ]

        # tasks = [
        #     self.embed_checkout_assistant_prompt(),
        #     self.embed_gmail_prompt(),
        #     self.embed_google_calendar_prompt(),
        #     self.embed_order_tracking_prompt(),
        #     self.embed_return_and_refund_prompt(),
        #     self.embed_shopping_cart_prompt(),
        #     self.embed_user_profile_management_prompt()
        #     ]
        # Wait for all tasks to complete
        loop.run_until_complete(asyncio.gather(*tasks))
        # Close the event loop
        loop.close()

        elapsed_time = time.time() - start
        print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    result = GenerateEmbeddingFile(brand_name='ATAS').run()
    print(result)