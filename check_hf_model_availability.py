from huggingface_hub import model_info

repo_id = "sentence-transformers/all-MiniLM-L6-v2"
info = model_info(repo_id)

print("Repository:", info.id)
print("Private:", info.private)
print("Gated:", info.gated)
