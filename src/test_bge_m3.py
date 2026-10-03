from FlagEmbedding import BGEM3FlagModel

print("Loading BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)

texts = [
    "I cannot log in to my account.",
    "The verification code is not working.",
    "I love this app."
]

print("Generating embeddings...")

output = model.encode(
    texts,
    batch_size=2,
    max_length=128
)

dense_vectors = output["dense_vecs"]

print("Embedding shape:", dense_vectors.shape)

print("\nTest completed successfully!")