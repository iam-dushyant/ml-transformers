from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [

    "PyTorch is a machine-learning framework.",

    "Transformers can create text embeddings.",

]

embeddings = model.encode(texts)

print(embeddings.shape)
