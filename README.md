# MLM_inconsistencies
Data and code for exposing the inconsistencies of conditionals learned by masked language models.

# Abstract


Learning to predict masked tokens in a sequence has been shown to be a powerful pretraining objective for large-scale language models. After training, such masked language models can provide distributions of tokens conditioned on bidirectional context. In this short draft, we show that such bidirectional conditionals often demonstrate considerable inconsistencies, i.e., they can not be derived from a coherent joint distribution when considered together. We empirically quantify such inconsistencies in the simple scenario of bigrams for two common styles of masked language models: T5-style and BERT-style. For example, we show that T5 models often confuse its own preference regarding two similar bigrams. Such inconsistencies may represent a theoretical pitfall for the research work on sampling sequences based on the bidirectional conditionals learned by BERT-style MLMs. This phenomenon also means that T5-style MLMs capable of infilling will generate discrepant results depending on how much masking is given, which may represent a particular trust issue.

# Code & data
python 3.11 

conda create --name my_inconsistencies python=3.11.5
conda activate my_inconsistencies
pip install -r requirements.txt




# Discussion

We are doing more experiments on this topic at the moment. Leave a comment under ''issues'' for questions/discussion.








# Strategy for different punctuations

click to expand
In the LAMBADA last word prediction task, natural language models (LLMs) may append various punctuations to the same last word, leading to different completions. For example, to complete the sentence "My color of my pet dog is":

Possible Completions:

white. with probability p_1
white! with probability p_2 (assuming p_1 > p_2)
black, with probability p_3
black? with probability p_4 (assuming p_3 > p_4)
Strategies to Rank white and black:

Maximum Probability Strategy
Probability of white: p(white) = p_1
Probability of black: p(black) = p_3
Sum of Probabilities Strategy
Probability of white: p(white) = p_1 + p_2
Probability of black: p(black) = p_3 + p_4
Afterwards p(_white_) and p(_black_) may need normalization.

WE ARE STICKING WITH MAXIMUM PROBABILITY STRAGEGY ACCORDING TO ACCURACIES OBTAINED FROM TRIAL RUNS.