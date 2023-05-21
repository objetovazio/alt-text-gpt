import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

from transformers import logging
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

sentence1 = "A young boy is playing with a hockey stick"
sentence2 = "a dog playing with a ball"

inputs1 = tokenizer(sentence1, return_tensors='pt')
inputs2 = tokenizer(sentence2, return_tensors='pt')

with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

vectors1 = outputs1.pooler_output
vectors2 = outputs2.pooler_output

similarity = cosine_similarity(vectors1, vectors2)
print(similarity)