from fastapi import FastAPI, File, UploadFile , Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
import fitz 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import faiss
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import openai
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain_community.chains import ConversationRetrivalChain



app = FastAPI()
app.conversation = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_pdf_text(filename ,file):
    doc = fitz.open(file.filename)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    text = '\n'.join(line for line in text.splitlines() if line.strip())

    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    load_dotenv()
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

@app.post("/process_text")
async def process_text(request: Request):
    raw_body = await request.body()
    text = raw_body.decode()
    json_data = json.loads(text)  
    print(json_data.get("question"))
    processed_text = text.upper()  # Example processing, converting text to uppercase
    return {"processed_text": processed_text}


@app.post("/chat")
async def chat(request: Request):
    # return app.counter 
    try:
        raw_body = await request.body()
        text = raw_body.decode()
        json_data = json.loads(text)  
        question = json_data.get("question")
        response = app.conversation.invoke({"question" : question})
        chat_history = response['chat_history']
        print(chat_history[-1].content)
        return chat_history
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": str(e)}, status_code=500)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            app.conversation=None
            load_dotenv()
            # get raw text
            raw_text=get_pdf_text(file.filename , file)

            #get text chunks
            text_chunks=get_text_chunks(raw_text)

            # get vectorstore
            vectorstore=get_vectorstore(text_chunks)
            
            app.conversation = get_conversation_chain(vectorstore)

        return JSONResponse(content={"message":"Success"}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": str(e)}, status_code=500)
