
from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
import pandas as pd


def carregar_documentos(csv_path):

    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    documentos = []
    for _, row in df.iterrows():
        texto = f"PERGUNTA: {row['Pergunta']}\nRESPOSTA: {row['Resposta']}"
        documentos.append(Document(page_content=texto))

    return documentos


def retrieve_info(query, db):
    
    similar_response = db.similarity_search(query, k=3)

    return [doc.page_content for doc in similar_response]


if __name__ == '__main__':

    llm_model = 'llama3.1'

    model = ChatOllama(model=llm_model, temperature=0)

    # loader = CSVLoader(file_path='base_conhecimento.csv')
    # documentos = loader.load()

    documentos = carregar_documentos('base_conhecimento.csv')

    embeddings = OllamaEmbeddings(model=llm_model)

    db = FAISS.from_documents(documentos, embeddings)
    db.save_local('db')



    template='''
    Você é um especialista em saúde pública com longa experiência no tratamento e prevenção da dengue.
    Responda à pergunta abaixo com base exclusivamente nas informações fornecidas na base de dados.

    Pergunta:
    {message}

    Informações relevantes extraídas do banco de dados:
    {best_practice}

    Resposta objetiva e clara para o cidadão:
    '''


    prompt = PromptTemplate(
        input_variables=['message', 'best_practice'],
        template=template,
    )

    chain = LLMChain(llm=model, prompt=prompt)



    message = 'Quando surgem os sintomas?'
    message = 'Como é transmissão do vírus da dengue'
    

    best_practice = retrieve_info(message, db)

    response = chain.run(message=message, best_practice=best_practice)
    
    print(response)




    # Acessar os documentos armazenados no índice
    documentos = db.docstore._dict.values()

    # Exibir o conteúdo dos documentos
    for doc in documentos:
        print(doc.page_content)

    for doc in documentos:
        print(doc.metadata)





    







