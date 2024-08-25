import os
import time
import fitz
import pickle
from common.chunking_constant import ChunkingStrategy, CHUNKING_CONFIG_PATH
from common.common_utils import read_json_file, write_pickle_file
from common.common_utils import remove_images_and_links_from_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.rag_models import Chunk


class ChunkingService:
    def __init__(self, config_file_path=CHUNKING_CONFIG_PATH):
        self.config_data = self._load_config(config_file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config_data["chunk_size"],
            chunk_overlap=self.config_data["chunk_overlap"],
            length_function=len,
            separators=self.config_data["separators"]
        )

    def _load_config(self, config_file_path):
        return read_json_file(config_file_path)

    def create_chunks_from_docs(self, document_path, save_chunks=True, dest_dir="data"):
        start_time = time.time()

        chunks_with_page_num = self._convert_pdf_to_text_chunks(document_path)

        chunk_objects = []
        for i, chunk_with_page_num in enumerate(chunks_with_page_num):
            chunk_objects.append(Chunk(id=i, page_num=chunk_with_page_num[0], chunk_text=chunk_with_page_num[1]))

        if save_chunks:
            write_pickle_file(os.path.join(dest_dir, "chunks.pkl"), chunk_objects)

        end_time = time.time()
        print(f"Time Elapsed for Chunking: {end_time - start_time} seconds")

        return chunk_objects

    def _convert_pdf_to_text_chunks(self, pdf_path):
        doc = fitz.open(pdf_path)
        print(f"Processing {pdf_path}")

        page_texts = {}
        chunks_with_page_num = []

        for page_num, page_content in enumerate(doc):
            text = page_content.get_text()

            text = remove_images_and_links_from_text(text)

            page_texts[page_num] = text  # page_num + 1 to start page numbering from 1

            page_chunks = self.convert_text_to_chunks(text)

            for page_chunk in page_chunks:
                chunks_with_page_num.append([page_num + 1, page_chunk])

        print(f"Total Pages Processed: {len(page_texts)}")

        return chunks_with_page_num

    def convert_text_to_chunks(self, text):
        return self.text_splitter.split_text(text)
