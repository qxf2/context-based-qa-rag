"""
A context-based question answering script that reads the context from a dataset
and answers questions based on it
"""
import os
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, \
                         DPRContextEncoderTokenizer, \
                         RagTokenizer, \
                         RagRetriever, \
                         RagSequenceForGeneration, \
                         logging

torch.set_grad_enabled(False)

# Suppress Warnings
logging.set_verbosity_error()

# Initialize context encoder & decoder model
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

dataset_name = "rony/soccer-dialogues"
localfile_name = dataset_name.split('/')[-1]

# load 100 rows from the dataset
ds = load_dataset(dataset_name, split='train[:100]')
def transforms(examples):
    """
    Transform dataset to be passed
    as an input to the RAG model
    """
    inputs = {}
    inputs['text'] = examples['text'].replace('_',' ')
    inputs['embeddings'] = ctx_encoder(**ctx_tokenizer(inputs['text'], return_tensors="pt"))[0][0].numpy()
    inputs['title'] = 'soccer'
    return inputs
ds = ds.map(transforms)

# Add faiss index to the dataset, it is needed for DPR
ds.add_faiss_index(column='embeddings')

# Initialize retriever and model
rag_model = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(rag_model)
retriever = RagRetriever.from_pretrained(rag_model, indexed_dataset=ds)
model = RagSequenceForGeneration.from_pretrained(rag_model, retriever=retriever)

# Generate output for questions
question = "How old is Granit Xhaka"
input_dict = tokenizer(question, return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=50)

print(f"{question}?")
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
