import os
from enum import Enum

EMBEDDING_CONFIG_PATH = os.path.join("embedding_service",
                                     "resources", "embedding_config.json")

MODEL_TYPE = "model_type"
EMBEDDING_MODELS = "embedding_models"


# Define the EmbeddingModelType enum
class EmbeddingModelType(Enum):
    OPEN_AI = "OPEN_AI"
    BAAI = "BAAI"


EMBEDDING_UPSERT_SIZE_IN_SINGLE_REQUEST = 20
