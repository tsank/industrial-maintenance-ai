# Equipment Manual Retrieval-Augmented Generation (RAG) System
# Builds the RAG chain: retriever + prompt + LLM + output parser
# Uses LCEL pipe syntax for clean composition
# Returns grounded answers with source citations

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env', override=True)


CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
COLLECTION_NAME = "maintenance_manuals"

PROMPT_TEMPLATE = """
You are a technical assistant for industrial equipment maintenance. 
Answer the technician's question using ONLY the provided manual excerpts.
If the answer is not in the excerpts, say excactly:
"This information is not available in teh loaded manuals."

Always end your answer with:
Source: [mention the relevant section or page if visible in the excerpts]

Manual Excerpts:
{context}

Technician question: {question}

Answer:
"""

def get_vectorstore():
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=int(CHROMA_PORT)
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Excerpt {i+1} | File: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def build_rag_chain():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_rag_chain()
    test_query = "What are the safety precautions for this equipment?"
    print(f"Test Query: {test_query}")
    print(f"Answer: {chain.invoke(test_query)}")
