from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

# Ajoute cette section avant la définition de l'app
app = FastAPI()

# Configuration CORS pour autoriser toutes les origines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines, ou spécifie ['http://localhost:8080'] par exemple
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permet tous les headers
)

# Reste du code ici


# Charger les variables d'environnement
load_dotenv()

# Définir la clé API et autres variables
api_key = os.getenv("API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_KEY")

# 1. Créer l'agent LangChain
memory = MemorySaver()
search = TavilySearchResults(max_results=2)  # Ajout de l'outil de recherche
tools = [search]

# Utiliser OpenAI avec le modèle GPT-3.5-turbo
model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc125"}}
# 2. Créer l'application FastAPI

# 3. Créer une classe pour les requêtes
class Message(BaseModel):
    message: str

# 4. Route pour discuter avec l'agent
@app.post("/chat")
async def chat_with_agent(request: Message):
    user_message = request.message
    
    # Créer un message à envoyer à l'agent
    response = agent_executor.invoke({"text": user_message})
    
    # Renvoyer la réponse de l'agent
    return {"response": response["messages"][-1].content}

# Lancer le serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
