from dotenv import load_dotenv
from langchain.messages import HumanMessage,AIMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from langchain.tools import tool

load_dotenv()
agent = create_agent(
    model=init_chat_model(model = "gemini-3.6-flash", model_provider = "google-genai"),
    tools=[])


response = agent.invoke(
    {
        "messages" : [
            HumanMessage(content="How are you today?"),
        ]
    }
)


print(response["messages"][-1].content[0]["text"])

load_dotenv()