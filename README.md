# MLM_inconsistencies
Data and code for exposing the inconsistencies of conditionals learned by masked language models.

Abstract：Learning to predict masked tokens in a sequence has been shown to be a powerful pretraining objective for large-scale language models. After training, such masked language models can provide distributions of tokens conditioned on bidirectional context. In this short draft, we show that such bidirectional conditionals often demonstrate considerable inconsistencies, i.e., they can not be derived from a coherent joint distribution when considered together. We empirically quantify such inconsistencies in the simple scenario of bigrams for two common styles of masked language models: T5-style and BERT-style. For example, we show that T5 models often confuse its own preference regarding two similar bigrams. Such inconsistencies may represent a theoretical pitfall for the research work on sampling sequences based on the bidirectional conditionals learned by BERT-style MLMs. This phenomenon also means that T5-style MLMs capable of infilling will generate discrepant results depending on how much masking is given, which may represent a particular trust issue.

pkls/acceptable_alternatives_bigrams.pkl: data used to expose and calculate the inconsistencies among the conditionals learned by Roberta. There are 4 bigrams for each context. The keys represent their ids in the original c4 dataset.

roberta_for_bigram_inconsistencies_in_bulk.py: code used to expose and calculate the inconsistencies among the conditionals learned by Roberta

acceptable_alternatives_1000_ignore_cws_nos_50.pkl: data used to expose and calculate the inconsistencies among the conditionals learned by T5

calculate_inconsistencies_t5_c4.py: code used to expose and calculate the inconsistencies among the conditionals learned by Roberta

We are doing more experiments on this topic at the moment. Leave a comment under ''issues'' for questions/discussion.

