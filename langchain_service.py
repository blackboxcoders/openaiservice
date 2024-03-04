from flask import Flask, request
from download_embedding import download_embedding
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores  import FAISS
import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

app = Flask(__name__)

#OPENAI_API_KEY = "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"
OPENAI_API_KEY = " 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"


os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY 

@app.route("/getresponsegpt", methods = ["GET"])
def getResponseGpt():

    user_prompt = request.args.get("user_prompt")
    embeddingGenerator = OpenAIEmbeddings()
    download_embedding()
    PATH_VECTORSTORE = "vectorStore"
    baseconocimiento = FAISS.load_local(PATH_VECTORSTORE+"/faiss_index",  embeddingGenerator )
    docs = baseconocimiento.similarity_search(user_prompt)
    
    template = """""
    Eres un chatbot que esta conversando con un humano.

    Dadas las siguientes partes extraidas de un documento extenso, una historia y una pregunta, cree una
    respuesta final.

    {context}
    {chat_history}
    humano: {human_input}

    """

    prompt = PromptTemplate(
        input_variables= ["context", "chat_history","human_input"], template=template

    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    llm = ChatOpenAI()
    chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

    
    res = chain.run(input_documents=docs, human_input=user_prompt, return_only_outputs=True)


    return res





if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8003, debug = True)

