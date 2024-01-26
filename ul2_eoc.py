# %% [markdown]
# This code uses UL2 to 
# 
# (1) measure inconsistencies in its bidirectional conditionals; 
# 
# (2) improve llm inference with Emsemble of Conditionals.  
# 
# 

# %% [markdown]
# ### Imports and global utils

# %%
'''imports'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import utils.general_utils as general_utils
# clear GPU memory
if False:   
    general_utils.kill_gpu_process(os.environ["CUDA_VISIBLE_DEVICES"])
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
import numpy as np
import pickle
import time
from tqdm import tqdm
import json
import utils.lambada_utils as lambada_utils
from utils.lambada_utils import LambadaProcessor


# %% [markdown]
# ### Load tokenizer and model

# %%
# We are using custom huggingface cache dirs in case the default one doesn't have the capacity, since the models can be quite large.
MY_HUGGINGFACE_CACHE_DIR ='huggingface_cache' # relative to this notebook path
tokenizer = AutoTokenizer.from_pretrained("google/ul2",
                                        cache_dir = MY_HUGGINGFACE_CACHE_DIR+'/google-ul2')


# %%
RUN_CELL = 1 # Load model 0
# device_map = general_utils.get_ul2_device_map('0,1')
device_map="balanced"
if RUN_CELL:
    model = T5ForConditionalGeneration.from_pretrained("google/ul2", 
                                                    cache_dir=MY_HUGGINGFACE_CACHE_DIR + '/google-ul2', 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch.bfloat16,
                                                    device_map=device_map)

# %%
RUN_CELL = 0 # Load model 1
# device_map=general_utils.get_ul2_device_map('2,3')
if RUN_CELL:
    model1 = T5ForConditionalGeneration.from_pretrained("google/ul2",
                                                        cache_dir=MY_HUGGINGFACE_CACHE_DIR + '/google-ul2',
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map='cuda:1')

# %% [markdown]
# ### Ensemble of Conditionals

# %%
'''instantiate the lambada processor'''
IS_DEVELOPMENT = True # Set to False to run on the test set
set_partition = 'validation_' if IS_DEVELOPMENT else '' # filename part for saving results

LAMBADA_TEST_DATA_PATH = "data/jsonls/validation.jsonl" if IS_DEVELOPMENT else "data/jsonls/test.jsonl"
UL2_MODE = "[NLG]"
processor = LambadaProcessor(tokenizer, ul2_mode=UL2_MODE, lambada_dataset_path=LAMBADA_TEST_DATA_PATH)
lambada = processor.dataset

ce_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #reduction='avg'
ce_loss_sum = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum') #reduction='sum'

# %% [markdown]
# Strategy for different punctuations
# <details>
# <summary>click to expand</summary>
# 
# In the LAMBADA last word prediction task, natural language models (LLMs) may append various punctuations to the same last word, leading to different completions. For example, to complete the sentence "My color of my pet dog is":
# 
# Possible Completions:
# 
# 1. _white._ with probability `p_1`
# 2. _white!_ with probability `p_2` (assuming `p_1 > p_2`)
# 3. _black,_ with probability `p_3`
# 4. _black?_ with probability `p_4` (assuming `p_3 > p_4`)
# 
# Strategies to Rank _white_ and _black_:
# 
# 1. Maximum Probability Strategy
# 
# - Probability of _white_: `p(white) = p_1`
# - Probability of _black_: `p(black) = p_3`
# 
# 2. Sum of Probabilities Strategy
# 
# - Probability of _white_: `p(white) = p_1 + p_2`
# - Probability of _black_: `p(black) = p_3 + p_4`
# 
# Afterwards `p(_white_)` and `p(_black_)` may need normalization.

# %%
RUN_CELL = 0 # '''Generate the top completions (through beam search) for each example, and get the word from each completion.'''
if RUN_CELL:
    # generate for all examples, and then get the words from the completions, and compare the first one with the target
    count_correct = 0 # No. correct last word predictions if only the top completion is considered
    count_correct_top_num_beams = 0 # ... if the top num_beams completions are considered
    count_no_words_found = 0  # No. examples where no valid last word is found

    # punctuated_word: the last word and the punctuation that follows it
    id_to_punctuated_words = {} # maps example index to a list of word and punc pairs; every punc is kept for each word
    id_to_punctuated_words_unique = {} # ...; every punc is kept for each word  
    id_to_completions_ids = {}

    MAX_COMPLETION_LENGTH = 8 # for last word prediction, 8 is sufficient
    NUM_BEAMS = 20 # 20 is sufficient; more doesn't help

    # for example_index in tqdm(range(10)): # len(lambada)
    for example_index in tqdm(range(len(lambada))): # len(lambada)
        input_string = lambada[example_index]['inputs_pretokenized']
        inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs,
                                max_length=MAX_COMPLETION_LENGTH, 
                                num_beams=NUM_BEAMS, 
                                num_return_sequences=NUM_BEAMS, 
                                output_scores=True,
                                eos_token_id=tokenizer.convert_tokens_to_ids('<extra_id_1>'), 
                                return_dict_in_generate=True)
        
        completions = [tokenizer.decode(outputs['sequences'][i]) for i in range(NUM_BEAMS)]
        completions_ids = [
            outputs['sequences'][i].cpu()
            for i in range(NUM_BEAMS)
            if processor.get_word_from_completion(completions[i]) is not None # if the completion has a valid last word
        ]

        words = processor.get_words_from_completions(completions)

        # TODO: combine them and move to utils.py
        completions_without_pad = processor.remove_pad_id(completions_ids)
        completions_without_pad_before_punctution = processor.before_first_punc(completions_without_pad)
        
        
        if words:
            if words[0] == lambada[example_index]['targets_pretokenized'][0]:
                count_correct += 1
        else:
            count_no_words_found += 1
            # print("no words found")
        punctuated_words = processor.get_punctuated_words(completions)
        id_to_punctuated_words[example_index] = punctuated_words
        words_unique = list(set(words))
        id_to_punctuated_words_unique[example_index] = []
        
        id_to_completions_ids[example_index] = completions_without_pad_before_punctution

        # find the best punctuatuation for each unique word (Maximum Probability Strategy, 
        # completions are naturally ordered by probs by generate()) TODO: move this for loop to utils.py
        for word in words_unique:
            found = 0
            # iterate through the word and punc pairs, and find the one that matches the word
            for punctuated_word in punctuated_words:
                # it is a match if pair = word + punc
                ENDING_PUNCTUATIONS = ',!.:;?'
                for punc in ENDING_PUNCTUATIONS:
                    if punctuated_word == word + punc:
                        id_to_punctuated_words_unique[example_index].append(punctuated_word)
                        found = 1
                        break
                if found == 1:
                    break
        
        # calculate the number of correct top num_beams: if the correct word is in the top num_beams, then it is correct
        for word in words_unique:
            if word == lambada[example_index]['targets_pretokenized'][0]:
                count_correct_top_num_beams += 1
                break
    print("count_correct", count_correct)
    # count_correct, NLU: 0.7595
    # count_correct, NLG: 0.7680
    # count_correct, S2S: 0.3743 (could be because how the mode handles extra_ids)


# %%
RUN_CELL = 0 # '''Save the beam search results by generate()'''
if RUN_CELL:
    timed_pickle_filename = 'data/pkls/' + set_partition + UL2_MODE + '_ul2_lambada_vanilla_beam_search_results_' + general_utils.get_time() + '.pickle'
    print(timed_pickle_filename)

    data_keys = ['count_correct', 'count_correct_top_num_beams', 'count_no_words_found',
                'id_to_punctuated_words', 'id_to_punctuated_words_unique', 'id_to_completions_ids']
    data = {}
    for key in data_keys:
        data[key] = locals()[key]

    with open(timed_pickle_filename, 'wb') as fp:
        pickle.dump(data, fp)

# %%
'''Load the beam search results'''
# timed_pickle_filename = 'data/pkls/ul2_lambada_vanilla_beam_search_results_2023-11-11 20:08:17.pickle'
timed_pickle_filename = 'data/pkls/validation_[NLG]_ul2_lambada_vanilla_beam_search_results_2023-11-29-22:45:19.pickle'

with open(timed_pickle_filename, 'rb') as fp:
    ul2_lambada_vanilla_beam_search_results = pickle.load(fp)
id_to_completions_ids = ul2_lambada_vanilla_beam_search_results['id_to_completions_ids']

# %% [markdown]
# K-offset Ensemble
# <details>
# <summary>Click to expand</summary>
# 
# __K-offset Ensemble__ is a particular type of __Ensemble of Conditionals__ for last word prediction tasks like lambada.
# 
# It aims to augment the only conditional distribution obtained by masking the last word with more distributions. The new distributions are obtained by masking the last __offset__ + 1 words.
# 
# An example with the _lambada[0]_
# 
# _lambada[0]['input_pretokenized']_: `... his mouth curved in a confident grin , i do n't care about <last_word>`
# 
# We consider candidates `['angels.', 'signs.', 'that.']`.
# 
# The baseline approach is to input `... his mouth curved in a confident grin , i do n't care about <extra_id_0>` to UL2 and obtain the distribution containing the 3 candidates.
# 
# For the offset=1 case in K-offset Ensemble, we mask an extra token `about` in the end and input instead
# 
# `... his mouth curved in a confident grin , i do n't care <extra_id_1>`
# 
# This gives us a different distribution regarding `['about angels.', 'about signs.', 'about that.']`. They are given in an autoregressive manner
# e.g., `p(about angels) = p(about) * p(angels|about)`. Therefore we will use conditionals in the style of `p(angels|about)` to augment the baseline conditionals.
# 
# Cases where __K__ is larger can be similarly derived.
# 
# 
# 

# %%
MAX_OFFSET = 15

# %%
RUN_CELL = 0 # '''Generate the offset samples'''
if RUN_CELL:
    id_and_offset_to_inputs_and_completions = \
        processor.get_offset_samples(
            ul2_lambada_vanilla_beam_search_results['id_to_completions_ids'], 
            max_offset=MAX_OFFSET,
            to_gpu=True
        )

# %%
RUN_CELL = 0 # Save the offset samples
if RUN_CELL:    
    timed_pickle_filename = 'data/pkls/offset_samples_' + set_partition + 'max_offset_' + str(MAX_OFFSET) + '_' + general_utils.get_time() + '.pickle'
    print(timed_pickle_filename)
    with open(timed_pickle_filename, 'wb') as fp:
        pickle.dump(id_and_offset_to_inputs_and_completions, fp)

# %%
RUN_CELL = 1 # Load the offset samples
if RUN_CELL:
    # timed_pickle_filename = 'data/pkls/offset_samples_parallel_max_offset_5_2023-11-21-20:01:12.pickle'
    timed_pickle_filename = 'data/pkls/offset_samples_validation_max_offset_15_2023-11-30-00:27:00.pickle'
    with open(timed_pickle_filename, 'rb') as fp:
        id_and_offset_to_inputs_and_completions = pickle.load(fp)

# %%
RUN_CELL = 1 # Obtain the avg_log_p_map_offset
if RUN_CELL:
# id_and_offset_to_input_and_completions:
# (id, offset) -> input_ids, [completion_ids_0, completion_ids_1, completion_ids_2,...]
    avg_log_p_map_offset = dict() # (id, offset, completion_index) -> avg_log_p of the tokens constituting the last word (might be punctuated)
    
    for example_index in tqdm(range(len(lambada))): 
    # for example_index in tqdm(range(1)): 
        if len(id_to_completions_ids[example_index]) == 0:
            continue
        for offset in range(MAX_OFFSET):
            completions_batch = id_and_offset_to_inputs_and_completions[(example_index, offset)]['labels']
            input_ids = id_and_offset_to_inputs_and_completions[(example_index, offset)]['inputs'].unsqueeze(0)
            outputs = lambada_utils.multi_labels_forward(model, input_ids, completions_batch)

            for completion_index in range(len(id_to_completions_ids[example_index])):
                avg_log_p = -ce_loss(
                    # Only care about the tokens corresponding to the last word and omit offset tokens 
                    # the first one is <extra_id_0> and omitted
                    outputs.logits[completion_index][1+offset:], 
                    completions_batch[completion_index][1+offset:]
                )
                avg_log_p_map_offset[(example_index, offset, completion_index)] = \
                    avg_log_p.detach().cpu().tolist()

# %%
RUN_CELL = 1 # Save the avg_log_p_map_offset
if RUN_CELL:
    pickle_filename = 'data/pkls/avg_log_p_map_' + set_partition + 'max_offset_' + str(MAX_OFFSET) + '_' + general_utils.get_time() + '.pickle'
    print(pickle_filename)
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(avg_log_p_map_offset, handle)

# %%
RUN_CELL = 0 # Load the avg_log_p_map for the offset samples
if RUN_CELL:
    pickle_filename = 'data/pkls/avg_log_p_map_max_offset_30_2023-11-15-08:30:28.pickle'
    # pickle_filename = 'data/pkls/avg_log_p_map_max_offset_5_2023-11-15-04:12:17.pickle'
    # avg_log_p_map_offset (Dict): (id, offset, completion_index) -> avg_log_p of the tokens constituting the last word (might be punctuated)
    with open(pickle_filename, 'rb') as handle:
        avg_log_p_map_offset = pickle.load(handle)

# %%
RUN_CELL = 0 # '''Max reduction to emsemble the K different conditionals for the same last word, i.e., only the maximum avg_log_p is kept for each last word across different offsets. 
# We test K-offset ensemble for K up to MAX_OFFSET_TEST; MAX_OFFSET_TEST should be <= MAX_OFFSET used during avg_log_p_map generation
if RUN_CELL:
    MAX_OFFSET_TEST = 9 
    offset_to_accuracy = dict()
    offset_to_correct_ids = dict()
    for offset_test in range(MAX_OFFSET_TEST + 1):
        count_correct = 0 # No. correct last word predictions with K-offset
        # Get the best completion based on avg_log_p_map_offset
        for example_index in tqdm(range(len(lambada))): # len(lambada)
            # Create a list of tuples (avg_log_p, completion) for each completion
            avg_log_p_and_completion = [
                (avg_log_p_map_offset[(example_index, offset, completion_index)], id_to_completions_ids[example_index][completion_index])
                for offset in range(offset_test + 1)
                for completion_index in range(len(id_to_completions_ids[example_index]))
            ]
            if len(avg_log_p_and_completion) == 0:
                continue
            # Find the tuple with the maximum avg_log_p; this is essentially max reduction
            best_avg_log_p, best_completion = max(avg_log_p_and_completion, key=lambda x: x[0])
            if processor.is_correct_completion(example_index, best_completion):
                count_correct += 1
                offset_to_correct_ids.setdefault(offset_test, []).append(example_index)
        offset_to_accuracy[offset_test] = count_correct / (len(lambada))
    print(offset_to_accuracy)

# %%
RUN_CELL = 0 # Quantify disagreement on last word predictions among K-offset conditionals
if RUN_CELL: 
    for NUM_CONDITIONALS in range(2, 6): # 2, 3, 4, 5; how many sets of conditionals to consider; offset = 0 and offset = 1 are 2 different sets of conditionals
        id_offset_to_lastword = dict()
        id_to_lastwords_by_offsets = dict()
        for offset in range(NUM_CONDITIONALS): # if NUM_CONDITIONALS = 2, then offset = 0, 1
            for example_index in range(len(lambada)): # len(lambada)
                # Create a list of tuples (avg_log_p, completion) for each completion
                avg_log_p_and_completion = [
                    (avg_log_p_map_offset[(example_index, offset, completion_index)], id_to_completions_ids[example_index][completion_index])
                    for completion_index in range(len(id_to_completions_ids[example_index]))
                ]
                if len(avg_log_p_and_completion) == 0:
                    continue
                # Find the tuple with the maximum avg_log_p; this is essentially max reduction
                best_avg_log_p, best_completion = max(avg_log_p_and_completion, key=lambda x: x[0])
                lastword = processor.get_word_from_completion(tokenizer.decode(best_completion))
                id_offset_to_lastword[(example_index, offset)] = lastword
                if example_index not in id_to_lastwords_by_offsets:
                    id_to_lastwords_by_offsets[example_index] = []
                id_to_lastwords_by_offsets[example_index].append(lastword)
        no_disagreement_count = 0
        for example_index in id_to_lastwords_by_offsets:
            if len(set(id_to_lastwords_by_offsets[example_index])) > 1:
                no_disagreement_count += 1
        ratio_disagreement = no_disagreement_count / (len(lambada) - ul2_lambada_vanilla_beam_search_results['count_no_words_found'])
        print("NUM_CONDITIONALS", NUM_CONDITIONALS, "ratio_disagreement", ratio_disagreement)

# %% [markdown]
# Middle-off ensemble (incomplete)
# <details>
# <summary>Click to expand</summary>
# 
# __Middle-off Ensemble__ is a particular type of __Ensemble of Conditionals__ for last word prediction tasks like lambada.
# 
# 
# It aims to augment the only conditional distribution obtained by masking some additional words in the middle of the input for additional distributions. The new distributions are obtained by masking the last __offset__ + 1 words.
# 
# The key sample generation function is create_middle_off_sample() in lambada_utils, which is controlled by
# `middle_span_length`: the length of the masked span in the middle
# and 
# `middle_to_end_gap`： the gap between the middle_span and the last word
# 
# 
# An example with the _lambada[0]_
# 
# _lambada[0]['input_pretokenized']_: `... his mouth curved in a confident grin , i do n't care about <last_word>`
# 
# We consider candidates `['angels.', 'signs.', 'that.']`.
# 
# The baseline approach is to input `... his mouth curved in a confident grin , i do n't care about <extra_id_0>` to UL2 and obtain the distribution containing the 3 candidates.
# 
# 
# 
# completion_lengths = [
#     id_and_offset_to_inputs_and_completions[example_index,0][completion_index][1].shape[0] - 1
#     for example_index in range(len(lambada)) 
#     for completion_index in range(len(id_and_offset_to_inputs_and_completions[example_index,0]))
# ] 
# np.mean(completion_lengths) == 3.8
# 

# %%
# define range_middle_span_length and range_middle_to_end_gap
RANGE_MIDDLE_SPAN_LENGTH = [3]
RANGE_MIDDLE_TO_END_GAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LENGTH_GAP_TUPLES = [
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), 
    (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)
]

# %%
RUN_CELL = 1 # Generate the middle-off samples
if RUN_CELL:
    # id_middlespan_gap_to_input_and_completions: maps (id, middle_span_length, middle_to_end_gap) to a input_ids(Tensor) and completion_ids(List[Tensor])
    id_middlespan_gap_to_input_and_completions = \
        processor.get_middle_off_samples(
            id_to_completions_ids, 
            range_middle_span_length=RANGE_MIDDLE_SPAN_LENGTH,
            range_middle_to_end_gap=RANGE_MIDDLE_TO_END_GAP,
            to_gpu=True
        )

# %%
RUN_CELL = 1 # Save the middle-off samples
if RUN_CELL:    
    timed_pickle_filename = 'data/pkls/middle_off_samples_' + set_partition + 'rmsl_' + str(RANGE_MIDDLE_SPAN_LENGTH[0]) + \
        '_rmteg_1_10' + '_' + general_utils.get_time() + '.pickle'
    print(timed_pickle_filename)
    with open(timed_pickle_filename, 'wb') as fp:
        pickle.dump(id_middlespan_gap_to_input_and_completions, fp)

# %%
RUN_CELL = 0 # '''Load the middle-off samples'''
if RUN_CELL:
    timed_pickle_filename = 'data/pkls/middle_off_samples_rmsl_3_rmteg_1_10_2023-11-25-20:48:49.pickle'
    with open(timed_pickle_filename, 'rb') as fp:
        id_middlespan_gap_to_input_and_completions = pickle.load(fp)

# %%
RUN_CELL = 1 # Obtain and save the avg_log_p_map for middle-off samples
if RUN_CELL:
    # id_middlespan_gap_to_input_and_completions: maps (id, middle_span_length, middle_to_end_gap) to a input_ids(Tensor) and completion_ids(List[Tensor])
    # avg_log_p_map_middle_off: maps (id, middle_span_length, middle_to_end_gap, completion_index) to avg_log_p of the tokens constituting the last word (might be punctuated)
    avg_log_p_map_middle_off = dict()
    for id_middlespan_gap in tqdm(id_middlespan_gap_to_input_and_completions):
        input_ids = id_middlespan_gap_to_input_and_completions[id_middlespan_gap]['inputs'].unsqueeze(0)
        completions_batch = id_middlespan_gap_to_input_and_completions[id_middlespan_gap]['labels']
        outputs = lambada_utils.multi_labels_forward(model, input_ids, completions_batch)

        middlespan_length = id_middlespan_gap[1]

        for completion_index in range(len(completions_batch)):
            avg_log_p = -ce_loss(
                # Only care about the tokens corresponding to the last word
                outputs.logits[completion_index][2+middlespan_length:], 
                completions_batch[completion_index][2+middlespan_length:]
            )
            avg_log_p_map_middle_off[(*id_middlespan_gap, completion_index)] = \
                avg_log_p.detach().cpu().tolist()
            
    '''Save the avg_log_p_map_middle_off'''
    pickle_filename = 'data/pkls/avg_log_p_map_middle_off_' + set_partition + 'rmsl_' + str(RANGE_MIDDLE_SPAN_LENGTH[0]) + \
        '_rmteg_1_10' + '_' + general_utils.get_time() + '.pickle'
    print(pickle_filename)
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(avg_log_p_map_middle_off, handle)
exit()

# %%
RUN_CELL = 1 # Load the avg_log_p_map_middle_off
if RUN_CELL:
    pickle_filename = 'data/pkls/avg_log_p_map_middle_off_rmsl_3_rmteg_1_10_2023-11-25-22:45:36.pickle'
    with open(pickle_filename, 'rb') as handle:
        avg_log_p_map_middle_off = pickle.load(handle)

# %%
'''Max reduction to emsemble middle-off conditionals for the same last word, 
i.e., only the maximum avg_log_p is kept for each last word across different range_middle_span_length's and range_middle_to_end_gap's.
Possibly emsemble with the K-offset conditionals.'''


ADD_MIDDLE_OFF = True # ADD the middle-off ensemble to the list
ADD_BASELINE = True # ADD the baseline (offset = 0 from K-offset ensemble) to the list
ADD_K_OFFSET = False # ADD the whole K-offset ensemble to the list
MAX_OFFSET = 9
LENGTH_GAP_TUPLES =  [(3,5)]

count_correct = 0
correct_ids = []
for example_index in tqdm(range(len(lambada))): # len(lambada)
    # Create a list of tuples (avg_log_p, completion) for each completion
    avg_log_p_and_completion = []
    # add middle-off to the list
    if ADD_MIDDLE_OFF:
        avg_log_p_and_completion += [
            (avg_log_p_map_middle_off[(example_index, middle_span_length, middle_to_end_gap, completion_index)], id_to_completions_ids[example_index][completion_index])
            for middle_span_length, middle_to_end_gap in LENGTH_GAP_TUPLES
            for completion_index in range(len(id_to_completions_ids[example_index]))
        ]
    # add the baseline (offset = 0 from K-offset ensemble) to the list
    if ADD_BASELINE:
        avg_log_p_and_completion += [
            (avg_log_p_map_offset[(example_index, 0, completion_index)], id_to_completions_ids[example_index][completion_index])
            for completion_index in range(len(id_to_completions_ids[example_index]))
        ]
        
    # add the whole K-offset ensemble to the list
    if ADD_K_OFFSET:
        avg_log_p_and_completion += [
            (avg_log_p_map_offset[(example_index, offset, completion_index)], id_to_completions_ids[example_index][completion_index])
            for offset in range(1, MAX_OFFSET + 1)
            for completion_index in range(len(id_to_completions_ids[example_index]))
        ]

    if len(avg_log_p_and_completion) == 0: # if no completions are found
        continue
    # Find the tuple with the maximum avg_log_p; this is essentially max reduction
    best_avg_log_p, best_completion = max(avg_log_p_and_completion, key=lambda x: x[0])
    if processor.is_correct_completion(example_index, best_completion):
        count_correct += 1
        correct_ids.append(example_index)
print("count_correct:", count_correct)
print("accuracy:", count_correct / len(lambada))


# %% [markdown]
# ### Notes
# 
#  Hypothesis: conditionals based on the mask patterns used during pretraining are more powerful;
# 
#  just ensemble with one LENGTH_GAP_TUPLE == (3,5) leads to accuracy: 0.7814865127110421
# 
# 

# %%
LENGTH_GAP_TUPLES[:1]

# %%
avg_log_p_and_completion

# %%
avg_log_p_and_completion

# %%
offset_to_correct_ids[9].__len__()

# %%
where_k_offset_helps = set(offset_to_correct_ids[9]) - set(offset_to_correct_ids[0])

# %%
where_middle_off_helps = set(correct_ids) - set(offset_to_correct_ids[0])

# %%
where_k_offset_helps.__len__()

# %%
where_middle_off_helps.__len__()

# %%
where_k_offset_helps.intersection(where_middle_off_helps).__len__()

# %%
(where_k_offset_helps - where_middle_off_helps).__len__()

# %%
(where_middle_off_helps - where_k_offset_helps).__len__()

# %%
'''Obtain the avg_log_p_map for middle-off samples via data parallelism'''
RUN_CELL = 0
if RUN_CELL:
    from multiprocessing import Process
    import multiprocessing
    avg_log_p_map_middle_off = dict()
    # define the processing for each id_middlespan_gap example as a function and use threading to use 3 models in parallel
    def process(list_id_middlespan_gap, model_, device='cuda:0'):
        for id_middlespan_gap in tqdm(list_id_middlespan_gap):
            input_ids = id_middlespan_gap_to_input_and_completions[id_middlespan_gap]['inputs'].unsqueeze(0).to(device)
            completions_batch = id_middlespan_gap_to_input_and_completions[id_middlespan_gap]['labels'].to(device)
            outputs = lambada_utils.multi_labels_forward(model_, input_ids, completions_batch)

            middlespan_length = id_middlespan_gap[1]

            for completion_index in range(len(completions_batch)):
                avg_log_p = -ce_loss(
                    # Only care about the tokens corresponding to the last word and omit offset tokens 
                    # the first one is <extra_id_0> and omitted
                    outputs.logits[completion_index][2+middlespan_length:], 
                    completions_batch[completion_index][2+middlespan_length:]
                )
                avg_log_p_map_middle_off[(*id_middlespan_gap, completion_index)] = \
                    avg_log_p.detach().cpu().tolist()
            
    # run the above function in parallel
    import threading
    multiprocessing.set_start_method('spawn')

    all_id_middlespan_gaps = list(id_middlespan_gap_to_input_and_completions.keys())
    all_id_middlespan_gaps_0 = all_id_middlespan_gaps[:len(all_id_middlespan_gaps)//2]
    all_id_middlespan_gaps_1 = all_id_middlespan_gaps[len(all_id_middlespan_gaps)//2:]
    # all_id_middlespan_gaps_2 = all_id_middlespan_gaps[2*len(all_id_middlespan_gaps)//3:]

    t0 = Process(target=process, args=(all_id_middlespan_gaps_0, model, 'cuda:0'))
    t1 = Process(target=process, args=(all_id_middlespan_gaps_1, model1, 'cuda:2'))
    # t2 = threading.Thread(target=process, args=(all_id_middlespan_gaps_2, model2, 'cuda:4'))

    t0.start()
    t1.start()
    # t2.start()

# %% [markdown]
# ### End of main code

# %%
''' Plot ensembled conditionals vs accuracy'''
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load a nice font
font_path = '/usr/share/fonts/urw-base35/NimbusMonoPS-Italic.otf'
font_prop = fm.FontProperties(fname=font_path)

# offset = 0 corresponds to the baseline, which is no. ensembled conditionals = 1; adjust the offset by 1
no_ensembled_conditionals_to_accuracy = dict()
for offset in range(1, MAX_OFFSET_TEST+1):
    no_ensembled_conditionals_to_accuracy[offset] = offset_to_accuracy[offset-1]


max_line = plt.plot(list(no_ensembled_conditionals_to_accuracy.keys()), list(no_ensembled_conditionals_to_accuracy.values()), label='max')
plt.xlabel('No. ensembled conditionals', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
# the interval on x should be 10
plt.xticks(np.arange(10, max(list(no_ensembled_conditionals_to_accuracy.keys()))+1, 10))\
# add a tick at 1 on the x axis
plt.xticks(list(plt.xticks()[0]) + [1])

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# add a dot at each point
plt.scatter(list(no_ensembled_conditionals_to_accuracy.keys()), list(no_ensembled_conditionals_to_accuracy.values()))


# add a yellow horizontal line at y=offset_to_accuracy[0]
plt.axhline(y=no_ensembled_conditionals_to_accuracy[1], color='y', linestyle='--')
# add the word "baseline" at the end of the yellow line in the font of calibri
plt.text(48, no_ensembled_conditionals_to_accuracy[1] + 0.0002, 'baseline', fontproperties=font_prop, fontsize=13)

# # plot the accuracy with avg reduction
# avg_line = plt.plot([item+1 for item in list(offset_to_accuracy_avg_reduction.keys())], list(offset_to_accuracy_avg_reduction.values()), color='r', label='avg')
# # add a dot at each point
# plt.scatter([item+1 for item in list(offset_to_accuracy_avg_reduction.keys())], list(offset_to_accuracy_avg_reduction.values()), color='r')

plt.scatter(1, no_ensembled_conditionals_to_accuracy[1], color='y')

plt.legend(handles=[max_line[0], avg_line[0]], loc='upper center', bbox_to_anchor=(0.9, 0.45), ncol=1, fontsize=10)


plt.tight_layout()

# show the plot at a high resolution
# plt.savefig('no_ensembled_conditionals_to_accuracy_combined.png', dpi=1200)

# plt.print()


# %%
import threading
import time

# A simple function that prints and sleeps
def print_numbers(name):
    for i in range(1, 6):
        time.sleep(2)
        print(f"{name} prints: {i}")

# Creating threads
thread1 = threading.Thread(target=print_numbers, args=("Thread 1",))
thread2 = threading.Thread(target=print_numbers, args=("Thread 2",))

# Starting threads
thread1.start()
thread2.start()

# Waiting for threads to complete
thread1.join()
thread2.join()

print("Threads finished execution")


# %%
import importlib
import utils.lambada_utils as lambada_utils  # Import the module, not just the class
importlib.reload(lambada_utils)
from utils.lambada_utils import LambadaProcessor  # Re-import the class

# %%
import importlib
import utils.general_utils as general_utils
importlib.reload(general_utils)

# %%
model.hf_device_map.keys().__len__()

# %%
general_utils.get_ul2_device_map('6,7').__len__()


