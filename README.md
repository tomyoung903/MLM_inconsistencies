# MLM_inconsistencies
Data and code for exposing the inconsistencies of conditionals learned by masked language models.
https://arxiv.org/abs/2301.00068

# Abstract


Learning to predict masked tokens in a sequence has been shown to be a powerful pretraining objective for large-scale language models. After training, such masked language models can provide distributions of tokens conditioned on bidirectional context. In this short draft, we show that such bidirectional conditionals often demonstrate considerable inconsistencies, i.e., they can not be derived from a coherent joint distribution when considered together. We empirically quantify such inconsistencies in the simple scenario of bigrams for two common styles of masked language models: T5-style and BERT-style. For example, we show that T5 models often confuse its own preference regarding two similar bigrams. Such inconsistencies may represent a theoretical pitfall for the research work on sampling sequences based on the bidirectional conditionals learned by BERT-style MLMs. This phenomenon also means that T5-style MLMs capable of infilling will generate discrepant results depending on how much masking is given, which may represent a particular trust issue.

# Code & data
python 3.11 

# Discussion

We are doing more experiments on this topic at the moment. Leave a comment under ''issues'' for questions/discussion.

