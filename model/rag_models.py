class Chunk:
  def __init__(self, id, page_num, chunk_text):
    self.id = id
    self.chunk_text = chunk_text
    self.page_num = page_num

  def __repr__(self):
    return f"<Chunk(id={self.id}, chunk_text={self.chunk_text}, page_ids={self.page_num})>"


class Embedding:
  def __init__(self, vector, chunk_text, page_num, id: int = None):
    self.id = id
    self.vector = vector
    self.chunk_text = chunk_text
    self.page_num = page_num
