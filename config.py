#########################################
#           PARAMETERS                  #
#########################################
TOP_K = 5 
MODEL = "nomic-ai/nomic-embed-text-v1.5"
TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"

DEVICE = "cpu" 

PERSIST_DIR = './storage/chroma_db/'

LLM_MODEL = None 
LOCAL_MODEL = None 

#########################################
#           STATE MANAGEMENT            #
#########################################
CURRENT_PDF = None  # holds currently loaded pdf 