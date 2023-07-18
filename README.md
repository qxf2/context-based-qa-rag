# Context-based Question Answering using RAG #

#### Why this post? ####
When I transitioned to the R&D department, I made the deliberate choice to tackle a recurring issue we consistently encountered at my previous client: support constantly bombarding engineers with queries that could have easily been resolved by referring to the release notes or troubleshooting guides. This prompted me to consider an alternative approach: What if support could seek assistance from someone else, allowing engineers to focus on engineering a high-quality product? This led me to explore the potential of leveraging Language Models (LLMs) to solve this problem. With recent advancements in tools for utilizing LLMs, I saw this as an invaluable opportunity to delve into their capabilities, further expanding my knowledge and expertise.
After exploring several other tools, I ultimately found myself working with <a href="https://arxiv.org/pdf/2005.11401.pdf">Retrieval Augmented Generation (RAG)</a>. In this post, my goal is to assist fellow engineers by addressing the issue I encountered when I first started working on this task - the challenge of finding reference materials that utilize non-technical definitions and language to explain concepts. Additionally, I intend for this post to serve as documentation of my progress in exploring context-based question answering

#### Tools of the trade: ####
In order to facilitate my exploration:
1. I selected a familiar context to work with: football. To achieve this, I opted to utilize the <a href="https://huggingface.co/datasets/rony/soccer-dialogues">rony/soccer-dialogues</a> dataset. Although this dataset is not directly related to the specific problem I aimed to solve, I intentionally chose it due to my extensive knowledge and understanding of football. This decision enabled me to engage with a subject matter that I was well-versed in.
2. I have utilized <a href="https://huggingface.co/docs/transformers/v4.17.0/en/index">Hugging Face's Transformers</a> module to download and use a pretrained model
3. I have used the <a href="https://huggingface.co/facebook/rag-sequence-nq">RAG-Sequence Model - facebook/rag-sequence-nq</a> model

#### Environment setup: ####
The setup process involves the following steps:

```
#Install Torch using the command:
pip install torch torchvision

#Install Transformers using the command:
pip install transformers

#Install Datasets using the command:
pip install datasets

#Install Faiss using the command:
pip install faiss-cpu
```

#### Unveiling the Logic: ####
The <em>facebook/rag-sequence-nq</em> model comprises three key components:
1. Question encoder
2. Retriever
3. Response generator
In this section, I will provide an in-depth explanation of each of these components, accompanied by illustrative code examples.

##### 1. Question encoder: ####
The Question encoder performs the task of encoding the question we want to pose to the LLM into a format that can be comprehended by the LLM.

```
from transformers import RagTokenizer

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

question = "How old is Granit Xhaka"
input_dict = tokenizer(question, return_tensors="pt")
```

##### 2. Retriever: #####
The Retriever component assists in locating a relevant context through its search capabilities.

```
from transformers import RagRetriever
from datasets import load_from_disk

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
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", indexed_dataset=ds)
```

##### 3. Response generator #####
The Response generator utilizes the context obtained from the Retriever component to generate output. It specifically responds to questions that have been encoded using the Question encoder.

```
from transformers import RagSequenceForGeneration

model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=50)
print(f"{question}? \n")
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

# Output
How old is Granit Xhaka? 
26 #If you are wondering why the output is 26, when Granit Xhaka is actually 30 years old(at the time of writing this post), this dataset is 4 years old.
```

##### Putting it all together: #####

```
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
```

#### Next steps: ####
With a functional prototype in hand, I am now committed to further exploration, aiming to gain a deeper understanding of the following aspects:
1. The inner workings of non-parametric memory generation models.
2. The utilization of DPR by RAG for retrieving relevant information.
3. Identifying effective methods for testing a RAG model.

