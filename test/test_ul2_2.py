from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

import os
os.environ["TRANSFORMERS_CACHE"] = "/work/09127/tomyoung/ls6/transformers_cache"


model = T5ForConditionalGeneration.from_pretrained("google/ul2", cache_dir='./ul2-cache', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
# .to("cuda")                                                                                                   
tokenizer = AutoTokenizer.from_pretrained("google/ul2")

input_string = "[NLU] Mr. Dursley was the director of a firm called <extra_id_0>, \
    which made <extra_id_1>. He was a big, solid man with a bald head. \
        Mrs. Dursley was thin and <extra_id_2> of neck, which came in very useful as she spent \
            so much of her time <extra_id_3>. The Dursleys had a small son \
                called Dudley and <extra_id_4>"                                               

print(sum(p.numel() for p in model.parameters()))
# 19,459,613,696
# inputs = tokenizer(input_string, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

# outputs = model.generate(inputs, max_length=200)

# print(tokenizer.decode(outputs[0]))
# -> "<pad><extra_id_0> Burrows<extra_id_1> a lot of money from the manufacture of a product called '' Burrows'''s ''<extra_id_2> had a lot<extra_id_3> looking down people's throats<extra_id_4> a daughter called Petunia. Dudley was a very stupid boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat, ugly boy who was always getting into trouble. He was a big, fat,"
