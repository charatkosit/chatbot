import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# ดึงค่าจาก environment variables
# openai_api_key = os.getenv('OPENAI_API_KEY')
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
# tavily_api_key = os.getenv('TAVILY_API_KEY')
# langchain_tracing_v2 = True
# langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
#commments
openai_api_key = "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"
anthropic_api_key = "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"
tavily_api_key = "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"
langchain_tracing_v2 = True
langchain_api_key = "ls__e8e668e392fd4d65a54e52b64c86cfa7"

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain_openai import ChatOpenAI, chat_models

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding)
    return vectorStore

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
    "input":user_input,
    "chat_history":chat_history
    })   
    return(response['output'])

# In-memory store for chat history
chat_histories = {}

app = Flask(__name__)
line_bot_api = LineBotApi('7GCEmFcjHYe0893CchShMcV/yh/b1ZFZgsn20/H+BB5mdnXpyBBNN6hCeTSqXcomnhbsHl22vJQKLPwbd+kxlCxwDCVb+xuZiePWuYOEzGidE9n/EYUXdnsyePoLJ9OOg1BKj++PpMCL0J8gOQ8fZwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('52841806b289e22472e45a372b82d04a')


def load_data():
    Dat1 = pd.read_csv(r'data/TOY-Table1.csv')
    Dat2 = pd.read_csv(r'data/TOY-Table2.csv')
    Dat2 = Dat2.drop_duplicates(subset=['PartID'])
    Parts = pd.merge(Dat1, Dat2, left_on='PartID', right_on='PartID', how='inner')
    Parts = Parts.drop(['PNC','Count','Catalogue','Fr-To','Model(Details)'], axis=1)
    Parts['PartID'] = Parts['PartID'].apply(lambda x: str(x).replace('-', ''))
    Parts['Car Brand'] = 'Toyota'
    Parts.columns = ['PartID', 'ItemName', 'Car Model', 'Car Brand']

    LoadStock = pd.read_csv(r'data/stock100k-r3.csv')
    LoadStock['ItemCode'] = LoadStock['ItemCodeNew'].apply(lambda x: str(x).split('-')[1])
    LoadStock = LoadStock.drop(['ID', 'ItemCodeNew', 'ItemName', 'Model'], axis=1)
    LoadStock.columns = ['Manufacturer', 'PartID', 'Price']
    LoadStock = LoadStock.drop_duplicates(subset=['PartID'])
    LoadStock = pd.merge(Parts, LoadStock, left_on='PartID', right_on='PartID', how='inner')

    LoadStock['ItemName'] = LoadStock['ItemName'] + ' ' + LoadStock['Car Brand'] + ' ' + LoadStock['Car Model'] + ' ' + LoadStock['PartID'] + ' ' + LoadStock['Price'].astype(str)
    LoadStock = LoadStock.drop(['Car Model', 'Car Brand', 'PartID', 'Price', 'Manufacturer'], axis=1)
    LoadStock.columns = ['Item Name']

    return LoadStock[:1000]

# Initialize Langchain components
from langchain_community.document_loaders import DataFrameLoader
def initialize_langchain():
    data = DataFrameLoader(load_data(), page_content_column='Item Name')
    Docs = data.load()

    vectorStore = create_db(Docs)
    retriever = vectorStore.as_retriever(search_kwargs={"k": 20})

    model = ChatOpenAI(
        model='gpt-4',
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a woman in auto parts distribution named G.P. \
         G.P. Auto Parts Company Limited was founded in 1981 with the \
         registered capital of 80 million baht. We are a comprehensive \
         leading automotive parts distributor and one of the most trusted distribution centers \
         who gained trust from clients such as parts shops, garages, motor insurances, fleets, \
         government and state enterprises, car clubs and end users with billions of sales per year. \
         Presently, the company has undergone a transformation to become a public company under the name GP \
         Mobility Public Company Limited since December 12, 2023. We are committed to providing excellent services with a vision \
         'Innovation driven leader in automotive aftermarket parts supply & service'. Our purpose is “Enhancing auto care and repair ecosystem"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    search = TavilySearchResults()
    retriever_tools = create_retriever_tool(
        retriever,
        "Parts_Search",
        "Only use when user needs data about auto parts and no data in chat history."
    )
    tools = [search, retriever_tools]

    agent = create_openai_functions_agent(
        llm=model,
        prompt=prompt,
        tools=tools
    )

    agentExecutor = AgentExecutor(
        agent=agent,
        tools=tools,
    )

    return agentExecutor

# Load Langchain components
agentExecutor = initialize_langchain()

# Flask route for Line webhook
@app.route("/callback", methods=['POST'])
def callback():
    # Get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # Get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    app.logger.info("X-Line-Signature header: " + signature)
    # Handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@app.route("/", methods=['GET'])
def home():
    return "Hello, this is the Flask app root."


# Handle messages from Line
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_message = event.message.text
    
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    if user_message =='clear':
        chat_histories[user_id] = []

    chat_history = chat_histories[user_id]
    response = process_chat(agentExecutor, user_message, chat_history)
    print(chat_history)
    #chat_histories[user_id].append({'user': user_message, 'bot': response})

    # Append the user message and bot response to the chat history
    chat_histories[user_id].append({'role': 'user', 'content': user_message})
    chat_histories[user_id].append({'role': 'assistant', 'content': response})


    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )

# # Process chat input
# def process_chat(agentExecutor, user_input):
#     chat_history = []
#     response = process_chat(agentExecutor,user_input,chat_history)
#     return response

if __name__ == '__main__':
    app.run(port=5001) 
