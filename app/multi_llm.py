import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dir = "app/text_store/ATAS_com"
# query = input("Enter Question: ")
query = 'help me to get refund for my purchases'
print("Command: ", query)

for txt_file in os.listdir(dir):
    print(txt_file)
    with open(os.path.join(dir,txt_file), "r") as file:
        
        document = file.read()
        print("Document:")
        # print(type(document))
        PROMPT = """Given an Instruction and a Document that describe about a assistant, you need to understand the capability and role of each assistant.
Based on your understanding, analyze if the assisstant is capable of carrying out the task.
If the assistant can excute the task, you answer yes. Else, you ansswer no.

<< Input Text >>
{{query}}

<< Document >>
{document}

<< Output >>
yes/no:
"""
        # print(PROMPT)
        
        # PROMPT = """
        # You are an examiner.
        # Given an Instruction and a Document that describe about a assistant, you need to understand the capability and role of each assistant.
        # Based on your understanding, analyze if the assisstant is capable of carrying out the task.
        # If the assistant can excute the task, you answer yes. Else, you ansswer no.
        
        # << Input Text >>
        # {{query}}

        # << Document >>
        # {document}

        # << Output >>
        # yes/no:
        # """

        
        chain = LLMChain(
            llm=ChatOpenAI(
                model = "gpt-3.5-turbo", 
                temperature=0.1),
            prompt=PromptTemplate.from_template(PROMPT)
        )
        result = chain.run(query)
        print(result)
        print()

