import torch
import transformers
import pandas as pd
import sklearn
import spacy
import networkx

print("🎉 Environment setup successful!")
print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Pandas version:", pd.__version__)
print("spaCy version:", spacy.__version__)
print("NetworkX version:", networkx.__version__)

print("\nDevice:", "CUDA GPU available 🚀" if torch.cuda.is_available() else "CPU mode 💻")