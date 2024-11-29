from fastapi import FastAPI
from transformers import pipeline


##Create a new FASTAPI instance
app = FastAPI()


##Initialise text genreation pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")   
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



@app.get("/")
def home():
    return{"message":"Hello World"}

##Define a function to get to handle get request at "/generate"

@app.get("/generate")
def generate(text:str):
    ##use pipeline to generate text from given input text
    output=pipe(text)

    ##return the text in generate Json response
    return{"output":output[0]['generated_text']}



