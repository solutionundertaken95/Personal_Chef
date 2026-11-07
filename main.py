from dotenv import load_dotenv
from langchain.messages import HumanMessage,AIMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from langchain.tools import tool
from tools import web_search
from prompt import CHEF_PROMPT, SYSTEM_CHEF_PROMPT
from photo_decoder import choose_image_file, build_image_message,image_to_data_url
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

load_dotenv()

logging.info("Creating Agent")
logging.info("Initializing Tavily client...")
agent = create_agent(
    model=init_chat_model(model = "gemini-3.6-flash", model_provider = "google-genai"),
    tools=[web_search],
    system_prompt=SYSTEM_CHEF_PROMPT)

logging.info("Image selection")
path = choose_image_file()
image_message = build_image_message(path, prompt=CHEF_PROMPT)

logging.info("Invoking agent with image message")
response = agent.invoke(
    {
        "messages" : [image_message]
    }
)


print(response["messages"][-1].content[0]["text"])

load_dotenv()