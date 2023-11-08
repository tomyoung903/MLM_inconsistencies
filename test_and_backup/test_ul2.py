from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os
import math
# os.environ["TRANSFORMERS_CACHE"] = "/work/09127/tomyoung/ls6/LLM_cache/models--google--ul2"

# cache_dir='./models--google--ul2',
model = T5ForConditionalGeneration.from_pretrained("google/ul2", cache_dir='/work/09127/tomyoung/ls6/LLM_cache/google-ul2/', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to("cuda")
model.parallelize()                                                                                                  
tokenizer = AutoTokenizer.from_pretrained("google/ul2")

input_string = "[NLU] Mr. Dursley was the director of a firm called <extra_id_0>, \
    which made <extra_id_1>. He was a big, solid man with a bald head. \
        Mrs. Dursley was thin and <extra_id_2> of neck, which came in very useful as she spent \
            so much of her time <extra_id_3>. The Dursleys had a small son \
                called Dudley and <extra_id_4>"                                           

inputs = tokenizer("[NLG] I will be having a <extra_id_0>", return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(inputs, output_scores=True, return_dict_in_generate=True, max_length=8)
outputs_beam = model.generate(inputs, num_beams=2, max_length=8, num_return_sequences=2, output_scores=True, return_dict_in_generate=True)
labels = tokenizer("<extra_id_0> sabbatical from", return_tensors="pt").input_ids.to("cuda")
labels = labels[:, :-1]
outputs_forward = model(input_ids=inputs, labels=labels)
loss_forward = outputs_forward.loss
logits_forward = outputs_forward.logits

# input_string = "[NLG] A man is having a bun for <extra_id_0>"
# inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
# num_beams = 1
# # outputs = model.generate(inputs, num_beams=num_beams, max_length=3, num_return_sequences=num_beams, output_scores=True, return_dict_in_generate=True)
# outputs = model.generate(inputs, max_length=3, output_scores=True, return_dict_in_generate=True)
# for i in range(num_beams):
#     # print(outputs['sequences'][i])
#     print(tokenizer.decode(outputs['sequences'][i]))
#     # decode outputs['sequences'][i] one by one token
#     print('-------------')
#     # print(outputs['sequences_scores'][i])




input_ids = tokenizer("[NLG] A man is having a bun for <extra_id_0>", return_tensors="pt").input_ids.to("cuda")
labels = tokenizer("<extra_id_0> lunch", return_tensors="pt").input_ids.to("cuda")
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

# recover likelihood from loss
likelihood = math.exp(-loss)
likelihood