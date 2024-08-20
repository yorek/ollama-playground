from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# make sure Ollama is running
embeddings = OllamaEmbeddings(
  model='mxbai-embed-large',
  show_progress=True
)

print("Loading text...")
with open("./ai-gov-executive-order.txt") as f:
    state_of_the_union = f.read()

print("Splitting...")
text_splitter = SemanticChunker(embeddings)
docs = text_splitter.create_documents([state_of_the_union])

print(f"Number of chunks {len(docs)}")

for i, d in enumerate(docs):
    print(f"Chuck {i:000}, Length: {len(d.page_content)}")

print("Chuck 0:")
print(docs[0].page_content)