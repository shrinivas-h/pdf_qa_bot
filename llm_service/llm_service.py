from llm_service.azure_open_ai_text_service import AzureTextModel
from llm_service.azure_openai_vision_service import AzureMultiModel
from llm_service.gemini_text_service import GeminiTextModel
from llm_service.gemini_vision_service import GeminiMultiModel


class LlmService:
    def __init__(self, llm_config):
        self.service_provider = llm_config.get('llm_provider')
        self.model_type = llm_config.get('model_type')
        self.details = llm_config.get('details')

    def get_response(self, context, query, list_of_image_paths=None):
        if list_of_image_paths is None:
            list_of_image_paths = []
        if self.service_provider == "AZURE":
            if self.model_type == "text_based":
                qa_model = AzureTextModel(self.details)
                result = qa_model.get_response(query, context)
                return result
            else:
                qa_model = AzureMultiModel(self.details)
                result = qa_model.get_response(list_of_image_paths, query, context)
                return result
        elif self.service_provider == "GCP":
            if self.model_type == "text_based":
                qa_model = GeminiTextModel(self.details)
                result = qa_model.get_response(query, context)
                return result
            else:
                qa_model = GeminiMultiModel(self.details)
                result = qa_model.get_response(list_of_image_paths, query, context)
                return result
