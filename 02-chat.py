from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  

chat_llm = AzureChatOpenAI(
    azure_deployment="sem-llm-test-gpt-35-turbo",
)

instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")

question = HumanMessage(content="What is the weather like?")

response = chat_llm.invoke([
    instructions,
    question
])

print(response.content)
