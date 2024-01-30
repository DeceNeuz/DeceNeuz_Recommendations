from fastapi import FastAPI
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification,TFAutoModel, pipeline
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()
loaded_tokenizer = AutoTokenizer.from_pretrained("base_model_tokenizer")
model_loaded_2 = TFAutoModelForSequenceClassification.from_pretrained("seq_model")
model_loaded_2_base = TFAutoModel.from_pretrained("seq_model")

# Make sure your payload has the news article text as articleText(given in class Article)
class Article(BaseModel):
    articleText: str

def predict_hate_speech(article_text:str):
    classifier2 = pipeline('sentiment-analysis',model = model_loaded_2,tokenizer = loaded_tokenizer)
    result = classifier2(article_text)[0]['label']
    # If prediction is hate -> 1 and if not_hate -> 0
    if result == 'hate':
        return 1
    else:
        return 0
    
def generate_word_embeddings(article_text:str):
    tokens = loaded_tokenizer.encode(article_text, add_special_tokens=True, return_tensors="tf")
    outputs = model_loaded_2_base(tokens)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    embeddings_new = embeddings.numpy()
    return embeddings_new[0]

# article_text = "People who skydive are retards"
# article_text2 = "People who get see tigers are blessed."

@app.get("/")
def root():
    return {"message": "Hello world"}

@app.post("/hate_speech")
def hate_speech_endpoint(article: Article):
    hate_speech_result = predict_hate_speech(article.articleText)
    return {"prediction": hate_speech_result}

@app.post("/word_embedding")
def word_embedding_endpoint(article: Article):
    word_embeddings = generate_word_embeddings(article.articleText)
    print("Hello this is working")
    return {"embedding": word_embeddings.tolist()}


