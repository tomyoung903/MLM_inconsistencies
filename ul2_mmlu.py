# %%
#/**
#* @author chenyunan (chen.yunan_01@nus.edu.sg)
#* @version 0.1
#* @date 2023-12-04
#* @copyright Copyright (c) 2023 
#*/

# %% [markdown]
# #### Tom's update Jan 1, 2024
# 
# 
#  accuracy currently at 0.336
# 
# (1) removed is_correct_completion(); we now simply check the index
# 
# (2) removed <extra_id_1> and <eos_token> from cross_entropy calculation
# 
# (3) fixed typos
# 
# (4) for Yunan -- how to further improve the code & performance:
#     
#     (a) On many of the samples, the constructed completions can contain <unk>'s, for example, when there is a { symbol in the completion. 
#         {AND, OR}
#         get tokenized into
#         <unk> AND , ▁OR <unk> </s>
# 
#         Having <unk>'s can hurt performance. One possible solution: remove the symbols can lead to <unk>'s 
# 
#     (b) Consider better prompt designs. For example, some questions end with a question mark, e.g., 
#         Q: If the foot is abducted, it is moved in which direction?	
#         A: 1. Inward	2. Outward	3. Upward	4. Downward
#         To make it "smoother" for the LLM, we can modify the prompt to be,
#         Prompt: Question: If the foot is abducted, it is moved in which direction? Answer: 
#         
#         Also, we can check how llm_eval_harness and instruct_eval did it.

# %% [markdown]
# ### Imports and global utils

# %%
'''imports'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import utils.general_utils as general_utils
# clear GPU memory
if True:   
    general_utils.kill_gpu_process(os.environ["CUDA_VISIBLE_DEVICES"])
import torch
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5.0' # suppresses pydevd speed warnings
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
import numpy as np
import pickle
import time
from tqdm import tqdm
import json
import utils.lambada_utils as lambada_utils
from utils.lambada_utils import LambadaProcessor
from typing import Tuple, List

# %% [markdown]
# ### Load tokenizer and model

# %%
# We are using custom huggingface cache dirs in case the default one doesn't have the capacity, since the models can be quite large.
MY_HUGGINGFACE_CACHE_DIR ='/data/personal/nus-ytj/MLM_inconsistencies/huggingface_cache' # relative to this notebook path
tokenizer = AutoTokenizer.from_pretrained("google/ul2",
                                        cache_dir = MY_HUGGINGFACE_CACHE_DIR+'/google-ul2')

RUN_CELL = 1 # Load model 1
# device_map=general_utils.get_ul2_device_map('2,3')
if RUN_CELL:
    model = T5ForConditionalGeneration.from_pretrained("google/ul2",
                                                        cache_dir=MY_HUGGINGFACE_CACHE_DIR + '/google-ul2',
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='cuda:0')

# %% [markdown]
# ### import MMLU datasets

# %%
from datasets import load_dataset

SUBJECTS = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', \
            'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', \
            'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', \
            'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', \
            'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', \
            'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', \
            'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', \
            'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', \
            'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', \
            'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', \
            'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']

SUBJECTS = SUBJECTS[:10] # tom is only using one subject for testing


DATASET_PATH = os.path.join("lukaemon/mmlu")
MMLU_DATAS = [load_dataset(DATASET_PATH, sub) for sub in SUBJECTS]
INDEX = [i for i in range(len(SUBJECTS))]
NAMES_WITH_DATAS = zip(INDEX, SUBJECTS, MMLU_DATAS)

# %% [markdown]
# #### Test generation

# %%
''''Test generated completion against constructed completion'''
RUN_CELL = 0
example_id = 3
NAMES_WITH_DATAS = list(NAMES_WITH_DATAS)
data = NAMES_WITH_DATAS[0][2]
if RUN_CELL:
    for example_id in range(10):
        MAX_COMPLETION_LENGTH = 100
        NUM_BEAMS = 1
        example = data['test'][example_id]
        # print(data['test'])
        example_input = example['input']
        input_string = '[NLG] ' + example_input + ' <extra_id_0>'
        # print('\ninput_string:', input_string)

        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs,
                                max_length=MAX_COMPLETION_LENGTH, 
                                num_beams=NUM_BEAMS, 
                                num_return_sequences=NUM_BEAMS, 
                                output_scores=True,
                                # eos_token_id=tokenizer.convert_tokens_to_ids('<extra_id_1>'), 
                                return_dict_in_generate=True)

        # print('\ncompletion generated:')
        # print(tokenizer.decode(outputs[0][0]))
        # print(outputs[0][0])
        # print all tokens put in a long string
        tokens = [tokenizer.convert_ids_to_tokens([id_])[0] for id_ in outputs[0][0]]
        print('-------')
        print('\u2588'.join(tokens))

        key = example['target']
        # print('\ncompletion constructed:')
        completion_constructed = f"<extra_id_0> {example[key]}"
        print(completion_constructed)
        # print(tokenizer(completion_constructed, return_tensors="pt").input_ids)
        tokens = [tokenizer.convert_ids_to_tokens([id_])[0] for id_ in tokenizer(completion_constructed, return_tensors="pt").input_ids[0]]
        print('\u2588'.join(tokens))
            

# %% [markdown]
# #### Define Loss Function

# %%
# define loss
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #reduction='avg'
ce_loss_sum = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum') #reduction='sum'

# %%
extra_id_0 = torch.tensor([tokenizer.convert_tokens_to_ids("<extra_id_0>")])
extra_id_1 = torch.tensor([tokenizer.convert_tokens_to_ids("<extra_id_1>")])

# %% [markdown]
# #### Define Question prompt

# %%
import torch.nn.functional as F
UL2_MODE = "[NLG]"

def data_prompting(docs, tokenizer) -> Tuple:
    '''
        docs: DATA_SET[SUBJECTS_NAME], ex:MMLU[high_school_european_history]
        return: Tuple(input_ids, labels)

        input[example]: Question:<prompt> 
        label[example]: A. <choice1> B. <choice2> C. <choice3> D. <choice4>

        Todo: few-shot data prompting
    '''
    keys = ["A", "B", "C", "D"]
    key_to_index = {"A":0, "B":1, "C":2, "D":3}
    for doc in docs:
        input_ = UL2_MODE + " " + doc['input'] + " " + "<extra_id_0>"
        # print(input_)
        # completions = [f"<extra_id_0> {doc[key]} <extra_id_1>" for key in keys]
        completions = [f"<extra_id_0> {doc[key]}" for key in keys]
        # print(completions)
        label = key_to_index[doc['target']]
        
        input_ids = tokenizer(input_, return_tensors="pt").input_ids.to("cuda").clone().detach().requires_grad_(False)
        # label_id = tokenizer(label, return_tensors="pt").input_ids.to("cuda").clone().detach().requires_grad_(False)
        # completions_ids = [tokenizer(completion, return_tensors="pt").input_ids.to("cuda").clone().detach().requires_grad_(False)\
                                                                # for completion in completions]
        completions_ids = [tokenizer(completion, return_tensors="pt").input_ids.to("cuda").clone().detach()[:,:-1]\
                                                                for completion in completions] # remove <eos> token with [:,:-1]
        # print(completions_ids)
        # Assuming `max_length` is the maximum length you want to pad sequences to
        max_length = max(seq.size(1) for seq in completions_ids)

        # Note to Yunan: Please compress the following 2 code lines to remove one "pad" function call; Consult chatgpt or official doc for guidance on how to pad simply and effectively
        # Pad sequences to the common length
        padded_sequences = [F.pad(seq, (0, max_length - seq.size(1)), value=tokenizer.pad_token_id) for seq in completions_ids]

        # Use pad_sequence
        completions_ids_padded = torch.nn.utils.rnn.pad_sequence(padded_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

        completions_ids_padded = torch.squeeze(completions_ids_padded, dim = 1)
        yield input_ids, completions_ids_padded, label

# %%
IS_DEVELOPMENT = False
set_partition = 'validation' if IS_DEVELOPMENT else 'test' 

# %%
RUN_CELL = 1 # Obtain the avg_log_p_map_offset
TOTAL_CASE = 0
ACCURATE_CASE = 0

if RUN_CELL:
# id_and_offset_to_input_and_completions:
# (id, offset) -> input_ids, [completion_ids_0, completion_ids_1, completion_ids_2,...]
    avg_log_p_map_offset = dict() # (id, offset, completion_index) -> avg_log_p of the tokens constituting the last word (might be punctuated)
    
    for example_index in tqdm(range(len(INDEX))): 
    # for example_index in tqdm(range(2)):
        data = MMLU_DATAS[example_index]
        # print(SUBJECTS[example_index])

        gen = data_prompting(data[set_partition], tokenizer)

        for input_ids, completions_batch, label in gen:
            if input_ids.shape[1] <= 512:
                continue
            print(input_ids.shape)

            avg_log_p_and_completion = []
            outputs = lambada_utils.multi_labels_forward(model, input_ids, completions_batch)
            
            # print('new completion batch')
            for completion_index in range(len(completions_batch)):
                
                avg_log_p = -ce_loss(
                    # Only care about the tokens corresponding to the last word and omit offset tokens 
                    # the first one is <extra_id_0> and omitted
                    outputs.logits[completion_index][1:], 
                    completions_batch[completion_index][1:]
                )
                
                avg_log_p_map_offset[(example_index, 0, completion_index)] = \
                    avg_log_p.detach().cpu().tolist()
                
                avg_log_p_and_completion.append([avg_log_p.detach().cpu().tolist(), completion_index])
                
            best_avg_log_p, best_completion_index = max(avg_log_p_and_completion, key=lambda x: x[0])
            
            break

            if best_completion_index == label:
                ACCURATE_CASE += 1
            TOTAL_CASE += 1

# %%
# String found in file /home/nus-ytj/miniconda3/envs/inconsistencies/lib/python3.11/site-packages/transformers/tokenization_utils_base.py on line 3832: "Token indices sequence length is longer than the specified maximum sequence length "


# %%
len(files)

# %%
ACCURATE_CASE / TOTAL_CASE


