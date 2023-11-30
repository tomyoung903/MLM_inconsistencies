from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math


tokenizer = T5Tokenizer.from_pretrained("t5-11b")
model = T5ForConditionalGeneration.from_pretrained("t5-11b", cache_dir='./t5-11b-cache')
model.parallelize()
model.model_parallel = True
input_sentence = "The chicken <extra_id_0> is a common choice of food."
print(input_sentence)
input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')

labels_sentences = ["<extra_id_0> burger <extra_id_1>", \
                    "<extra_id_0> breast <extra_id_1>", \
                    "<extra_id_0> salad <extra_id_1>", \
                    "<extra_id_0> satay <extra_id_1>"]
probs = []
for i in range(len(labels_sentences)):
    labels_sentence = labels_sentences[i]
    print(labels_sentence)
    labels = tokenizer(labels_sentence, return_tensors="pt").input_ids
    labels = labels[:, :-1]
    print(tokenizer.convert_ids_to_tokens(labels[0], skip_special_tokens=False))
    labels = labels.to('cuda:0')
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    # recover likelihood from loss
    normalized_likelihood = math.exp(-loss)
    likelihood = normalized_likelihood ** labels.shape[1]
    probs.append(likelihood)
    print(likelihood)

