import os

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

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

result = movie_plot_vector.similarity_search("A movie where aliens land and attack earth.")
for doc in result:
    print(doc.metadata["title"], "-", doc.page_content)
