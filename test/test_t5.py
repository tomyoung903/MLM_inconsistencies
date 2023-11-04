from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("t5-11b")
model = T5ForConditionalGeneration.from_pretrained("t5-11b", cache_dir='/work/09127/tomyoung/ls6/LLM_cache/t5-11b-cache')
# model = T5ForConditionalGeneration.from_pretrained("./t5-large-cache")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
model.parallelize()
# model = model.module
input_sentence = "The chicken <extra_id_0> is a common choice of food."
print(input_sentence)
input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')
sequence_ids = model.generate(input_ids)

outputs = model.generate(input_ids, 
                        return_dict_in_generate=True, 
                        eos_token_id=tokenizer.convert_tokens_to_ids('<extra_id_1>'),
                        num_beams=100,
                        output_scores=True,
                        num_return_sequences=100)

for i in range(len(outputs.sequences)):
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(outputs.sequences[i], skip_special_tokens=False)))
    # print(outputs.scores[i])

# print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

# # create the model
# model_class = T5ForConditionalGeneration

# model = T5ForConditionalGeneration.from_pretrained("t5-11b", cache_dir='./t5-11b-cache')
# model.parallelize()
# # Initialize the DeepSpeed-Inference engine

# ds_engine = deepspeed.init_inference(model,
#                                  mp_size=2,
#                                  dtype=torch.float,
#                                  checkpoint=None,
#                                  replace_method='auto',
#                                  replace_with_kernel_inject=True)

# model = ds_engine.module
# output = model('Input String')