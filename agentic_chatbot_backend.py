from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-20b",
                 groq_api_key=os.getenv("GROQ_API_KEY"))

### Defining the State

class ChatState(TypedDict):

    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):

    ## take user query from state
    messages = state['messages']
    ## send to llm
    response = model.invoke(messages)
    ## return response
    return {'messages':[response]}


checkpoint = MemorySaver()

graph = StateGraph(ChatState)

## Add nodes.

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')

graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpoint)

