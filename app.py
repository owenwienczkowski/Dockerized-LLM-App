from fastapi import FastAPI, Body
from transformers import pipeline
from pydantic import BaseModel

# SAMPLE INPUT FOR TERMINAL: curl -X POST "http://localhost:5000/classify/" -H "Content-Type: application/json" -d "{\"text\":\"I love this movie!\"}"

app = FastAPI()

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

last_classification = None

class TextRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    input_text: str
    classification: str

@app.post("/classify/", response_model=ClassificationResponse)
async def classify(text_request: TextRequest):
    global last_classification
    # Use the sentiment analysis pipeline to classify the input text
    classification_result = sentiment_analyzer(text_request.text)[0]['label']
    print("Classification Result: " + classification_result)

    # Store the last classification result
    last_classification = classification_result

    return ClassificationResponse(input_text=text_request.text, classification=classification_result)

@app.get("/classify/")
async def get_classification():
    if last_classification is None:
        return {"message": "No classification has been made yet."}
    return {"last_classification": last_classification}

@app.get("/")
async def read_root():
    return {"message": "Send a POST request to /classify/ to classify text."}