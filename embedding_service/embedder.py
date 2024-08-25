import os
import time
from typing import List
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from chunking_service.chunker import ChunkingService
from common.embedding_constant import EMBEDDING_CONFIG_PATH, MODEL_TYPE, EMBEDDING_MODELS, EmbeddingModelType
from model.rag_models import Embedding, Chunk
from common.common_utils import read_json_file, write_pickle_file
from parallel_processor.parallel_processor import ParallelProcessorThreadPool


class EmbeddingService:
    def __init__(self, config_path=EMBEDDING_CONFIG_PATH):
        self.config_data = read_json_file(config_path)
        self.initialize_embedding_model()

    def initialize_embedding_model(self):
        model_type_str = self.config_data[MODEL_TYPE]

        model_type = EmbeddingModelType(model_type_str)

        if model_type.name in self.config_data[EMBEDDING_MODELS]:
            model_config = self.config_data[EMBEDDING_MODELS][model_type.name]
            self.model = HuggingFaceBgeEmbeddings(
                model_name=model_config["model_name"],
                model_kwargs=model_config.get("model_kwargs", {}),
                encode_kwargs=model_config.get("encode_kwargs", {})
            )
        else:
            raise ValueError(f"Model type {model_type.name} not found in configuration.")

    def find_embeddings_for_string(self, string):
        embedding = self.model.embed_query(string)
        return embedding

    def process_chunk(self, chunk: Chunk) -> Embedding:
        # Find embedding for the chunk text
        vector = self.find_embeddings_for_string(chunk.chunk_text)

        # Create an Embedding object with id, embedding, and page_num
        return Embedding(
            id=chunk.id,
            vector=vector,
            chunk_text=chunk.chunk_text,
            page_num=chunk.page_num
        )

    def embeddings_for_list_of_chunks(self, chunk_objects, parallel_processing=True):
        if parallel_processing:
            processor = ParallelProcessorThreadPool(self.process_chunk, num_threads=4, thread_name_prefix="embedding")

            # Prepare arguments for parallel processing
            chunk_args = [(chunk,) for chunk in chunk_objects]
            # Run the processing in parallel
            results = processor.run(*chunk_args)
        # Create an instance of the thread pool processor
        else:
            results = []
            # Iterate over the list of Chunk objects
            for chunk in chunk_objects:
                # Find embedding for the chunk text
                vector = self.find_embeddings_for_string(chunk.chunk_text)
                print(f"{chunk.id} -> {chunk.page_num}-> {vector[:5]}")
                # Create an Embedding object with id, embedding, and page_num
                embedding_obj = Embedding(
                    id=chunk.id,
                    vector=vector,
                    chunk_text=chunk.chunk_text,
                    page_num=chunk.page_num
                )
                results.append(embedding_obj)

        return results

    @staticmethod
    def get_cosine_similarity_score(embedding1, embedding2):
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def rank_paragraphs_by_similarity_score(self, para_embeddings_with_id_page_num, query_embedding):
        para_embeddings_and_score = []
        for embedding_object in para_embeddings_with_id_page_num:
            idx, page_num, para_embedding, page_text = embedding_object.id, embedding_object.page_num, embedding_object.vector, embedding_object.chunk_text
            score = self.get_cosine_similarity_score(para_embedding, query_embedding)
            para_embeddings_and_score.append([idx, page_num, page_text, para_embedding, score])

        return sorted(para_embeddings_and_score, key=lambda x: x[4], reverse=True)

    def generate_embedding_from_chunks(self, chunk_objects: List[Chunk], save_embeddings=True, dest_dir="data"):

        start_time = time.time()

        # Ensure that the model is initialized
        if not self.model:
            raise ValueError("Model not initialized. Call initialize_embedding_model first.")

        # Generate embeddings for the list of Chunk objects
        embedding_objects = self.embeddings_for_list_of_chunks(chunk_objects)

        end_time = time.time()
        print(f"Time Elapsed for Parallel Embedding: {end_time - start_time} seconds")

        if save_embeddings:
            write_pickle_file(os.path.join(dest_dir, "embeddings.pkl"), embedding_objects)

        return embedding_objects
