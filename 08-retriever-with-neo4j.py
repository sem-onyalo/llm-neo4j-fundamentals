import os

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

llm = AzureChatOpenAI(
    azure_deployment="sem-llm-test-gpt-35-turbo",
)

embedding_provider = AzureOpenAIEmbeddings(
    azure_deployment="sem-llm-test-text-embedding-ada-002",
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

prompt = hub.pull("rlm/rag-prompt")

qa_chain = (
    {
        "context": movie_plot_vector.as_retriever(),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = qa_chain.invoke("A movie where a mission to the moon goes wrong.")

print(response)
