import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

llm = AzureChatOpenAI(
    azure_deployment="sem-llm-test-gpt-35-turbo",
)

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

llm_chain = template | llm

response = llm_chain.invoke({"fruit": "apple"})

print(response)
