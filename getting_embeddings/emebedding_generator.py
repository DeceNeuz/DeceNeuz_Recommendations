from fastapi import FastAPI, HTTPException
import openai
from openai.embeddings_utils import get_embedding
# import os
# from dotenv import load_dotenv
from pydantic import BaseModel
from pinecone import Pinecone

# load_dotenv()

API_key = "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"

openai.api_key = API_key



# initialize connection to pinecone (get API key at app.pinecone.io)
pc = Pinecone(
    api_key="d8b8199a-5e8e-45c3-930d-a18a7fc80b98",
    environment="gcp-starter"
)

app = FastAPI()

class ArticleContent(BaseModel):
    id: int
    content: str

# cntnt = """'In a remarkable turn of events, scientists have made groundbreaking discoveries in the realm of astrophysics, unraveling the mysteries of the cosmos. Through advanced telescopes and cutting-edge technology, researchers have observed distant galaxies, shedding light on the formation of stars and galaxies. These findings challenge previous theories and open new frontiers for our understanding of the universe. Astrophysicists worldwide are collaborating on this cosmic journey, with implications for our understanding of dark matter, black holes, and the origins of the universe.'"""


def com_embedding(content):
    # cntnt = get_content_from_api()
    mbddng =  get_embedding(content,engine = 'text-embedding-ada-002')
    return mbddng

def store_in_pinecone(id,embed):
    index = pc.Index("content-based-recc")
    index.upsert(
    vectors=[
        {
            "id": id, 
            "values": embed
        }
    ],
    namespace= "ns1")



@app.post('/post_article')
def post_article(article_content: ArticleContent):
# def post_article():
    try:
        # 1. Access the article content from the frontend
        received_content = article_content.content
        received_id = str(article_content.id)
        # received_content = cntnt

        # 2. Use OpenAI model to generate article embedding (placeholder code)
        # Replace this with the actual code for calling the OpenAI model
        # For demonstration purposes, using a placeholder function
        generated_embedding = com_embedding(received_content)
        store_in_pinecone(received_id,generated_embedding)

        # 3. Send the generated embedding back to the frontend along with a success message
        return {"message": "Article submitted and stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




