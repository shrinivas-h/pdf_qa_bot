import os
from enum import Enum

# Chunking Config path definition
CHUNKING_CONFIG_PATH = os.path.join("chunking_service",
                                    "resources", "chunking_config.json")

# Image related configurations
IMG_SAVE_DIR = os.path.join("rag_pipeline", "chunking_service", "images")
IMG_EXTENSION = ".jpg"


# Strategy enum definition
class ChunkingStrategy(Enum):
    PDF_TO_TEXT = "pdf_to_text"
    PDF_TO_DOCX_TO_TEXT = "pdf_to_docx_to_text"
    LLM_SERVICE = "llm_service"


class ChunkingStrategyValue(Enum):
    PDF_TO_TEXT = 0
    PDF_TO_DOCX_TO_TEXT = 1
    LLM_SERVICE = 2


