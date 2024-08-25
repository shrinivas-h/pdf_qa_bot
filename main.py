from chunking_service.chunker import ChunkingService
from common.common_utils import read_json_file
from embedding_service.embedder import EmbeddingService
from llm_service.openai_service import OpenAIChatClient
from constants.ai_service_constants import DOCUMENT_PATH, MAX_PAGES, QUESTIONS_PATH, API_KEY, MODEL, SLACK_TOKEN, \
    SLACK_CHANNEL
from slack_service.slack_bot import SlackMessenger


class DocumentProcessor:
    def __init__(self, config_file):
        self.config_data = read_json_file(config_file)
        self.document_path = self.config_data[DOCUMENT_PATH]
        self.max_pages = self.config_data[MAX_PAGES]
        self.api_key = self.config_data[API_KEY]
        self.model = self.config_data[MODEL]
        self.queries = read_json_file(self.config_data[QUESTIONS_PATH])
        self.embedding_objects = None

        # Initialize services
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.chat_client = OpenAIChatClient(self.api_key, self.model)

    def process_documents(self):
        # Create chunks from documents
        chunk_objects = self.chunking_service.create_chunks_from_docs(self.document_path)
        return chunk_objects

    def generate_embeddings(self, chunk_objects):
        # Generate embeddings from chunks
        self.embedding_objects = self.embedding_service.generate_embedding_from_chunks(chunk_objects)

    def get_responses(self):
        responses = []
        for query_id in self.queries:
            query = self.queries[query_id]
            print("Query: ", query)
            # Find embeddings for the query
            query_embedding = self.embedding_service.find_embeddings_for_string(query)

            # Rank paragraphs by similarity score
            ranked_paragraphs = self.embedding_service.rank_paragraphs_by_similarity_score(
                self.embedding_objects, query_embedding
            )

            # Create context for the query
            context = "You are an assistant that helps extract and interpret information from PDF reports. The process involves reading the text from a PDF document and then answering specific questions based on that extracted text. For this task, you are provided with the text extracted from the PDF and a user question. Your goal is to use the context of the extracted text to provide a meaningful and accurate answer to the user's question. The output should strictly be in json format with question, answer, and snippet keys without deviation. If the answer is unavailable or unclear, output Data Not Available in answer key. "

            text = context + "\n\n"
            for ranked_paragraph in ranked_paragraphs[:self.max_pages]:
                text += ranked_paragraph[2] + "\n"

            # Get completion from the chat client
            answer = self.chat_client.get_completion(system_message=text, user_message=query)
            responses.append(answer)

        return responses

    def get_config(self):
        return self.config_data


def main():
    # Initialize DocumentProcessor with the configuration file
    processor = DocumentProcessor("resources/config.json")
    config_data = processor.get_config()

    # Process documents and generate embeddings
    chunks = processor.process_documents()
    processor.generate_embeddings(chunks)

    # Load queries and get responses
    responses = processor.get_responses()
    for i in range(len(responses)):
        if 'json' not in responses[i]:
            responses[i] = "json\n" + responses[i]

    # Print responses (or handle them as needed)
    messenger = SlackMessenger(token=config_data[SLACK_TOKEN],
                               channel_id=config_data[SLACK_CHANNEL])

    messenger.send_bulk_messages(responses)


if __name__ == "__main__":
    main()
