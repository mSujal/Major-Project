#########################################
#           PARAMETERS                  #
#########################################
TOP_K = 5 
MCQ_TOP_K = 10  # as test if more context helps...

MCQ_RELEVANCE_THRESHOLD = 0.8
MODEL = "nomic-ai/nomic-embed-text-v1.5"
TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"

DEVICE = "cpu" 

PERSIST_DIR = './storage/chroma_db/'

LLM_MODEL = "openai/gpt-oss-120b"   # Groq: llama-3.3-70b-versatile / 
LOCAL_MODEL = None           # ollama fallback ko lagi —  machine ma jun ollama model pulled xa tyo

#########################################
#           STATE MANAGEMENT            #
#########################################
CURRENT_PDF = None  # holds currently loaded pdf 

TAXONOMY_PATH = "/home/sujan/Major-Project/Datasets/Jsons/embededsystem.json"