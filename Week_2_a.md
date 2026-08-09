# Week 2_a — Understanding a (Pre-trained) Hugging Face Text Classification Model

## Purpose

The objective of this exercise is not simply to run a pre-trained model. The aim is to understand the complete inference path:

```text
raw text
  ↓
tokenisation
  ↓
token IDs + attention mask
  ↓
transformer model
  ↓
logits
  ↓
probabilities
  ↓
predicted class
```

The model used here is:

```text
distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

It is a DistilBERT model fine-tuned for binary sentiment classification.

The two output classes are:

```text
NEGATIVE
POSITIVE
```

---

# 1. Load the tokenizer

```python
from transformers import AutoTokenizer

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## What this code does

`AutoTokenizer.from_pretrained(...)` downloads or loads the tokenizer associated with the selected model.

A transformer model does not operate directly on Python strings. It operates on numerical tensors.

The tokenizer is therefore responsible for converting:

```text
"I love this movie."
```

into a numerical representation that the model understands.

## Why the tokenizer must match the model

The tokenizer and model share the same vocabulary and tokenisation rules.

For example, if the tokenizer assigns:

```text
"movie" → token ID 3185
```

then the model has been trained assuming that ID `3185` refers to that token.

Using an unrelated tokenizer would therefore produce IDs whose meaning does not match the model's learned parameters.

---

# 2. Tokenise a sentence

```python
text = "I love this movie."

tokens = tokenizer.tokenize(text)

print(tokens)
```

Typical output may look similar to:

```text
['i', 'love', 'this', 'movie', '.']
```

## First-principles definition: token

A **token** is one unit of text recognised by the tokenizer.

A token is not necessarily the same as a word.

Depending on the tokenizer, a word can be split into smaller parts.

Conceptually:

```text
unbelievable
```

could become something similar to:

```text
un
believ
able
```

The exact split depends on the tokenizer vocabulary.

The reason for tokenisation is practical: a neural network cannot perform numerical operations directly on arbitrary words or sentences.

---

# 3. Convert tokens to token IDs

```python
ids = tokenizer.encode(text)

print(ids)
```

Typical output:

```text
[101, 1045, 2293, 2023, 3185, 1012, 102]
```

The exact values are determined by the tokenizer vocabulary.

## First-principles definition: token ID

A **token ID** is an integer used as an index into the model vocabulary.

Conceptually:

```text
"i"      → 1045
"love"   → 2293
"this"   → 2023
"movie"  → 3185
```

These numbers are not:

- probabilities,
- importance scores,
- embeddings,
- sentiment values.

They are identifiers.

A useful engineering analogy is an index into a lookup table.

---

# 4. Convert IDs back to tokens

```python
print(tokenizer.convert_ids_to_tokens(ids))
```

This is useful for checking what the integer IDs represent.

The result may contain special tokens such as:

```text
[CLS]
[SEP]
```

or their tokenizer-specific equivalents.

These special tokens are added because the model was trained expecting them.

For a classification model, the input therefore contains both the original text tokens and model-specific structural tokens.

---

# 5. Create tensors for the model

```python
encoded = tokenizer(
    text,
    return_tensors="pt"
)

print(encoded)
```

The output is normally a dictionary containing values such as:

```text
input_ids
attention_mask
```

Example:

```text
{
    'input_ids': tensor([[ 101, 1045, 2293, 2023, 3185, 1012, 102 ]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])
}
```

---

# 6. First-principles definition: tensor

A **tensor** is a multi-dimensional array of numbers.

Examples:

Scalar:

```text
5
```

One-dimensional array:

```text
[5, 8, 2]
```

Two-dimensional array:

```text
[
    [5, 8, 2],
    [1, 4, 7]
]
```

PyTorch represents numerical data using tensors because neural-network operations are implemented as tensor operations.

For example:

```python
print(encoded["input_ids"].shape)
```

might return:

```text
torch.Size([1, 7])
```

This means:

```text
1 input sequence
7 token positions
```

The first dimension is usually the **batch dimension**.

The second dimension is the **sequence length**.

---

# 7. Why there are double brackets

The tensor:

```text
[[101, 1045, 2293, 2023, 3185, 1012, 102]]
```

contains one sequence inside a batch.

Conceptually:

```text
batch
└── sentence 1
    └── token IDs
```

If three sentences were processed together, the first dimension would normally become `3`.

For example:

```text
shape = [3, sequence_length]
```

Batch processing is important because models are designed to process multiple examples efficiently.

---

# 8. Attention mask

```python
print(encoded["attention_mask"])
```

For a single unpadded sentence, the result may be:

```text
tensor([[1, 1, 1, 1, 1, 1, 1]])
```

## First-principles definition: attention mask

The **attention mask** indicates which positions contain actual input tokens and which positions are padding.

A simplified interpretation is:

```text
1 = real token
0 = padding
```

This becomes easier to see when processing two sentences of different lengths.

```python
texts = [
    "I love this movie.",
    "Bad."
]

encoded_batch = tokenizer(
    texts,
    padding=True,
    return_tensors="pt"
)

print(encoded_batch["input_ids"])
print(encoded_batch["attention_mask"])
```

The shorter sentence must be padded so that both rows have the same length.

Conceptually:

```text
Sentence A:
[101, ..., 102]

Sentence B:
[101, ..., 102, PAD, PAD]
```

The corresponding mask may look like:

```text
Sentence A:
[1, 1, 1, 1, 1, 1]

Sentence B:
[1, 1, 1, 1, 0, 0]
```

The zeros tell the model that those padded positions should not be treated as meaningful input.

---

# 9. Load the classification model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

## Breaking down the class name

```text
Auto
Model
For
Sequence
Classification
```

### Auto

`Auto` means that Hugging Face reads the model configuration and chooses the correct underlying model architecture.

In this case, the architecture is based on DistilBERT.

### Model

This is the neural network itself.

### Sequence

The input is a sequence of tokens.

For this exercise, the sequence represents one sentence.

### Classification

The model is expected to output a class for the entire input sequence.

For this model, the classes are:

```text
NEGATIVE
POSITIVE
```

## Why not use a generic AutoModel?

A generic transformer model normally returns internal hidden representations.

`AutoModelForSequenceClassification` includes an additional classification head that converts the transformer's internal representation into class scores.

---

# 10. Model configuration

```python
print(model.config.id2label)
```

Typical output:

```text
{
    0: 'NEGATIVE',
    1: 'POSITIVE'
}
```

The model itself produces numerical outputs.

The configuration provides the mapping between numerical class indices and readable class labels.

---

# 11. Run inference

```python
import torch

with torch.no_grad():
    outputs = model(**encoded)
```

## What `model(**encoded)` means

`encoded` is a Python dictionary containing:

```text
input_ids
attention_mask
```

Using:

```python
model(**encoded)
```

passes those dictionary entries into the model as named arguments.

It is effectively similar to:

```python
model(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"]
)
```

---

# 12. Why use `torch.no_grad()`

```python
with torch.no_grad():
```

PyTorch normally tracks mathematical operations because this information is required to compute gradients during training.

During inference, no training is taking place.

Therefore gradient tracking is unnecessary.

Disabling it reduces unnecessary memory use and computation.

The important distinction is:

```text
training
→ forward pass
→ loss
→ gradients
→ parameter update
```

whereas inference is:

```text
forward pass
→ output
```

---

# 13. Inspect the model output

```python
print(outputs)
```

For a sequence-classification model, the important value is:

```python
outputs.logits
```

Example:

```text
tensor([[-2.4, 3.1]])
```

There are two values because this model has two output classes.

---

# 14. First-principles definition: logits

A **logit** is a raw numerical score produced by the final layer of a classification model.

For example:

```text
NEGATIVE → -2.4
POSITIVE →  3.1
```

The model therefore assigns a much higher score to the positive class.

However, logits are **not probabilities**.

They can:

- be negative,
- be greater than 1,
- have any real numerical value.

The important information is their relative magnitude.

---

# 15. Convert logits to probabilities

```python
logits = outputs.logits

probabilities = torch.softmax(logits, dim=-1)

print(probabilities)
```

Possible result:

```text
tensor([[0.004, 0.996]])
```

This can be interpreted as approximately:

```text
NEGATIVE → 0.4%
POSITIVE → 99.6%
```

---

# 16. First-principles definition: softmax

**Softmax** converts a set of arbitrary real-valued scores into values between `0` and `1` whose total is `1`.

Conceptually:

```text
raw scores
[-2.4, 3.1]

        ↓ softmax

probabilities
[0.004, 0.996]
```

Softmax does not change which class has the highest score.

It converts the raw scores into a more interpretable form.

---

# 17. Select the predicted class

```python
predicted_class = probabilities.argmax(dim=-1).item()

print(predicted_class)
```

`argmax` returns the index of the largest value.

For:

```text
[0.004, 0.996]
```

the largest value is at index `1`.

Therefore:

```text
predicted_class = 1
```

The configuration maps:

```text
1 → POSITIVE
```

So:

```python
label = model.config.id2label[predicted_class]

print(label)
```

returns:

```text
POSITIVE
```

---

# 18. Dropout

When inspecting the model architecture, layers such as the following may appear:

```text
Dropout(p=0.2)
```

## First-principles definition: dropout

**Dropout** is a regularisation technique mainly used during model training.

During training, a fraction of activations is randomly set to zero.

Conceptually:

```text
before dropout:
[0.8, 1.2, 0.3, 2.1, 0.7]

after dropout:
[0.8, 0.0, 0.3, 0.0, 0.7]
```

The intention is to reduce over-dependence on particular activations and improve generalisation.

During normal evaluation/inference, dropout is disabled when the model is in evaluation mode.

This distinction is important:

```text
training:
dropout active

inference:
dropout inactive
```

---

# 19. What the transformer is doing between input IDs and logits

The simplified path is:

```text
token IDs
    ↓
embedding lookup
    ↓
vectors representing each token
    ↓
transformer layers
    ↓
contextualised token representations
    ↓
classification head
    ↓
logits
```

The token IDs themselves are only integer identifiers.

Before meaningful neural-network computation can take place, those IDs are mapped to dense numerical vectors called embeddings.

The transformer then repeatedly updates those representations using attention and feed-forward layers.

By the end of the network, the representation contains contextual information extracted from the sequence.

The classification head converts that learned representation into one score per output class.

For this model:

```text
final representation
    ↓
classification layer
    ↓
2 logits
```

because there are two classes.

---

# 20. Embedding lookup: why token IDs are useful

Suppose:

```text
"movie" → token ID 3185
```

The model contains an embedding matrix.

Conceptually:

```text
embedding_table[3185]
```

returns a vector such as:

```text
[0.17, -0.43, 0.08, ..., 0.29]
```

The actual embedding contains many more dimensions.

This is the first point at which the simple integer token ID becomes a learned numerical representation.

The key distinction is:

```text
token ID
= vocabulary index

embedding
= learned vector representation
```

---

# 21. High-level Hugging Face pipeline

The same task can also be executed using:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model=model_name
)

result = classifier("I love this movie.")

print(result)
```

This produces a convenient result such as:

```text
[
    {
        'label': 'POSITIVE',
        'score': 0.99...
    }
]
```

The pipeline is convenient, but it hides several intermediate stages.

Conceptually:

```text
pipeline(text)
```

performs approximately:

```text
text
  ↓
tokenizer
  ↓
input IDs + attention mask
  ↓
model forward pass
  ↓
logits
  ↓
softmax
  ↓
label mapping
```

For learning purposes, manually reproducing these stages is more useful than relying only on `pipeline()`.

---

# 22. Minimal end-to-end implementation

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_name = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

text = "I love this movie."

encoded = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**encoded)

logits = outputs.logits

probabilities = torch.softmax(
    logits,
    dim=-1
)

predicted_class = probabilities.argmax(
    dim=-1
).item()

label = model.config.id2label[
    predicted_class
]

score = probabilities[0][predicted_class].item()

print("Input:", text)
print("Token IDs:", encoded["input_ids"])
print("Attention mask:", encoded["attention_mask"])
print("Logits:", logits)
print("Probabilities:", probabilities)
print("Prediction:", label)
print("Score:", score)
```

---

# 23. How I should explain this in an interview

A concise explanation would be:

> The input starts as raw text. The Hugging Face tokenizer converts the text into tokens and then integer token IDs that correspond to entries in the model vocabulary. Those IDs are stored in PyTorch tensors. An attention mask identifies which positions contain real tokens and which positions are only padding. The token IDs are passed into a pre-trained DistilBERT sequence-classification model. Internally, the IDs are converted into learned embeddings and processed by transformer layers. The classification head produces one raw score, or logit, for each class. Softmax converts those logits into probabilities, and the highest-probability class is mapped back to a readable label such as POSITIVE or NEGATIVE.

This explanation is more important than memorising the exact Hugging Face syntax.

---

# 24. Checks I should be able to perform without copying code

Before considering this exercise complete, I should be able to answer the following from memory.

### Tokenisation

What problem does tokenisation solve?

I should be able to explain why raw text has to be converted into numerical input before the model can process it.

### Token IDs

What does a token ID represent?

I should know that it is a vocabulary index, not a probability or embedding.

### Tensor shape

What does:

```text
[1, 7]
```

mean?

I should recognise this as one input sequence containing seven token positions.

### Attention mask

Why does the model need an attention mask?

I should be able to explain padding and why padded positions must not be treated as meaningful input.

### Sequence classification

Why use:

```python
AutoModelForSequenceClassification
```

instead of a generic model?

I should understand that this model includes a task-specific classification head.

### Logits

What is a logit?

I should be able to state that it is a raw class score and not a probability.

### Softmax

Why use softmax?

I should understand that it converts raw scores into a probability distribution.

### Dropout

Why is dropout present in the architecture?

I should know that it is primarily a training-time regularisation mechanism and is disabled during normal inference.

---

# 25. Experiments worth doing

The following small experiments are more useful than adding more code.

## Experiment 1 — inspect tokenisation

Try:

```text
I love this movie.
```

and:

```text
I absolutely loved the cinematography.
```

Compare the tokens and token IDs.

---

## Experiment 2 — inspect padding

Use:

```python
texts = [
    "Excellent.",
    "This was one of the best films I have watched recently."
]
```

Run the tokenizer with:

```python
padding=True
```

Inspect:

```text
input_ids
attention_mask
shape
```

The objective is to understand why zeros appear in the attention mask.

---

## Experiment 3 — inspect logits

Try clearly positive and clearly negative statements.

For example:

```text
This product is excellent.
```

and:

```text
This product is terrible.
```

Compare their logits before looking at the probabilities.

The objective is to see that the model first produces raw class scores.

---

## Experiment 4 — ambiguous input

Try:

```text
The product arrived yesterday.
```

The model will still return a sentiment classification.

This illustrates an important engineering point: a model can produce a confident output even when the input does not strongly match the intended task.

Model output should therefore not automatically be interpreted as ground truth.

---

# 26. Working definition of the complete inference path

The complete process can be summarised as:

```text
1. Raw text is provided.

2. The tokenizer splits the text into model-specific tokens.

3. Tokens are mapped to integer vocabulary IDs.

4. IDs are stored in tensors.

5. Padding may be added so that multiple sequences can be processed
   together.

6. The attention mask identifies real tokens and padded positions.

7. Token IDs are mapped to learned embedding vectors.

8. Transformer layers process those vectors in context.

9. The sequence-classification head produces one logit per class.

10. Softmax converts logits into probabilities.

11. The class with the largest probability is selected.

12. The class index is converted into a readable label.
```

---

# 27. What I should retain from Week 2

The main outcome is not the ability to call a Hugging Face API.

The useful engineering understanding is:

```text
raw text
≠ model input
```

A series of transformations is required:

```text
text
→ tokens
→ token IDs
→ tensors
→ embeddings
→ transformer computation
→ logits
→ probabilities
→ label
```

Hugging Face provides abstractions that make these steps convenient, but the underlying data flow is still important when debugging models, exporting them, using ONNX, integrating them into other runtimes, or explaining their behaviour in a technical interview.
