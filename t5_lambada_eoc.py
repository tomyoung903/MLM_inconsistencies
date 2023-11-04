# %%
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os
import numpy as np
import pickle
import time
from tqdm import tqdm

# %%
tokenizer = AutoTokenizer.from_pretrained("t5-3b")

# %%
model = T5ForConditionalGeneration.from_pretrained("t5-3b", cache_dir='/work/09127/tomyoung/ls6/inconsistencies_project/t5-3b-cache', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(torch.device("cuda:2"))

# %%
device = torch.device("cuda:2")

# %%
# define show as a function that takes any kind of input and does nothing
def show(*args, **kwargs):
    pass

# %%
show = print

# %%
# /work/09127/tomyoung/ls6/glm/GLM-130B/evaluation_data/evaluation/lambada/lambada/please_next_word/gen/test.jsonl
import json
import os
with open("/work/09127/tomyoung/ls6/glm/GLM-130B/evaluation_data/evaluation/lambada/lambada/please_next_word/gen/test.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]
# append [NLG] to the beginning of each input, and <extra_id_0> to the end
data_appended = [{"inputs_pretokenized": x['inputs_pretokenized'] + " <extra_id_0>", "targets_pretokenized": x['targets_pretokenized']} for x in data]

# %%
from string import punctuation as PUNCTUATIONS
PUNCTUATIONS_LIST = list(PUNCTUATIONS)
PUNCTUATIONS_LIST.remove("<")
PUNCTUATIONS_LIST.remove(">")
PUNCTUATIONS_LIST.remove("_")
PUNCTUATION_IDS_LIST = [tokenizer.get_vocab()[p] for p in PUNCTUATIONS_LIST if p in tokenizer.get_vocab()]
for i in range(len(PUNCTUATION_IDS_LIST)):
    print(PUNCTUATION_IDS_LIST[i])
    print(tokenizer.decode(PUNCTUATION_IDS_LIST[i]))


# %%
def get_words_from_options(options):
    '''Get the first word from each of the given options. Return the words.'''
    # if a punctuation can be found in the option, get the word before the punctuation
    words = []
    for option in options:
        # find the punctuation
        for i in range(len(option)):
            if option[i] in PUNCTUATIONS_LIST:
                word = option[:i]
                words.append(word)
                # print(words)
                break

    # if the word starts with <pad>, remove it
    words = [word[5:] if word.startswith("<pad>") else word for word in words]

    # check it it the case that, assert that if the word starts with <extra_id_0>, ' ' follows. print the word if it is not the case
    for word in words:
        if word.startswith("<extra_id_0>") and len(word) > 13:
            if word[12] != " ":
                print('word[12] != \" \"')
                print(word)

    # if the word starts with <extra_id_0>, remove it
    words = [word[12:] if word.startswith("<extra_id_0>") else word for word in words]
    # if the word starts with ' ', remove it
    words = [word[1:] if word.startswith(" ") else word for word in words]
    # if the word ends with ' ', remove it
    words = [word[:-1] if word.endswith(" ") else word for word in words]
    # if the word is empty, remove it
    words = [word for word in words if word != ""]
    # if there are multiple words in word, remove it
    words = [word for word in words if len(word.split(" ")) == 1]
    return words

# %%
def get_word_from_option(option):
    '''Get the first word from the given option. Return the word.'''
    found = False
    # if a punctuation can be found in the option, get the word before the punctuation
    for i in range(len(option)):
        if option[i] in PUNCTUATIONS_LIST:
            word = option[:i]
            found = True
            break
    if not found:
        return None

    # if the word starts with <pad>, remove it
    word = word[5:] if word.startswith("<pad>") else word

    # check it it the case that, assert that if the word starts with <extra_id_0>, ' ' follows. print the word if it is not the case
    if word.startswith("<extra_id_0>") and len(word) > 13:
        if word[12] != " ":
            show('word[12] != \" \"')
            show(word)

    # if the word starts with <extra_id_0>, remove it
    word = word[12:] if word.startswith("<extra_id_0>") else word
    # if the word starts with ' ', remove it
    word = word[1:] if word.startswith(" ") else word
    # if the word ends with ' ', remove it
    word = word[:-1] if word.endswith(" ") else word
    # if the word is empty, remove it
    word = word if word != "" else None
    # if there are multiple words in word, remove it
    if word:
        word = word if len(word.split(" ")) == 1 else None
    return word

# %%
def get_word_punc_pairs(options):
    '''given a list of options (completions by the LLM), return a list of word-punc pairs'''
    # show(options)
    # if a punctuation can be found in the option, get the word before the punctuation
    words = []
    for option in options:
        # find the punctuation
        for i in range(len(option)):
            if option[i] in PUNCTUATIONS_LIST:
                word = option[:i+1]
                words.append(word)
                # show(words)
                break
    
    # if the word starts with <pad>, remove the <pad>
    words = [word[5:] if word.startswith("<pad>") else word for word in words]
    # if the word starts with <extra_id_0>, remove the <extra_id_0>
    words = [word[12:] if word.startswith("<extra_id_0>") else word for word in words]
    # if the word starts with ' ', remove it
    words = [word[1:] if word.startswith(" ") else word for word in words]
    # if the word ends with ' ', remove it
    words = [word[:-1] if word.endswith(" ") else word for word in words]
    # if the word is empty, remove it
    words = [word for word in words if word != ""]
    # if there are multiple words in word, remove it
    words = [word for word in words if len(word.split(" ")) == 1]
    # if the length is 1, remove it (to prevent the case where it is just a punctuation)
    words = [word for word in words if len(word) > 1]
    # if the word contains <unk>, remove it
    words = [word for word in words if "<unk>" not in word]
    return list(set(words))

# %%
def remove_pad(options):
    '''given a list of options (completions by the LLM), remove the <pad>'''
    # if the word starts with <pad>, remove the <pad>
    options = [option[5:] if option.startswith("<pad>") else option for option in options]
    return options

# %%
def remove_pad_id(options):
    '''given a list of options of ids (completions by the LLM), remove the <pad>'''
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    # if the word starts with <pad>, remove the <pad>
    options_return = []
    for option in options:
        if option[0] == pad_id:
            options_return.append(option[1:])
        else:
            options_return.append(option)
    return options_return

# %%
def before_first_punc(options):
    options_return = []
    for option in options:
        for i in range(len(option)):
            if option[i] in PUNCTUATION_IDS_LIST:
                options_return.append(option[:i+1])
                break
    return options_return

# %%
from tqdm import tqdm
# cross entroy loss with logits and labels
import torch
import torch.nn as nn
import torch.nn.functional as F
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #reduction='sum'
# loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
loss_fn_sum = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum') #reduction='sum'
# loss

# %%
# generate for all examples, and then get the words from the options, and compare the first one with the target
count_correct = 0
count_correct_top_num_beams = 0
count_no_words_found = 0
id_to_word_and_punc_pairs = {}
id_to_word_and_punc_pairs_processed = {}
id_to_options = {}
for example_index in tqdm(range(len(data_appended))): # len(data_appended)
    input_string = data_appended[example_index]['inputs_pretokenized']
    inputs = tokenizer(input_string, return_tensors="pt").input_ids.to(device)
    num_beams = 50
    outputs = model.generate(inputs, 
                             max_length=8,
                             num_beams=num_beams, 
                             num_return_sequences=num_beams, 
                             output_scores=True,
                             eos_token_id=tokenizer.convert_tokens_to_ids('<extra_id_1>'), 
                             return_dict_in_generate=True)
    
    options = [tokenizer.decode(outputs['sequences'][i]) for i in range(num_beams)]
    # for option in options:
    #     print(option)
    # print([tokenizer.batch_decode(outputs['sequences'][i]) for i in range(num_beams)])
    options_ids = [outputs['sequences'][i] for i in range(num_beams) if get_word_from_option(options[i]) is not None]
    words = get_words_from_options(options)
    options_without_pad = remove_pad_id(options_ids)
    options_without_pad_before_punctution = before_first_punc(options_without_pad)
    # print(words)
    if words:
        # print(options)
        # print(words[0], data_appended[example_index]['targets_pretokenized'])
        if words[0] == data_appended[example_index]['targets_pretokenized'][0]:
            count_correct += 1
    else:
        count_no_words_found += 1
        print("no words found")
    word_and_punc_pairs = get_word_punc_pairs(options)
    id_to_word_and_punc_pairs[example_index] = word_and_punc_pairs
    words_unique = list(set(words))
    id_to_word_and_punc_pairs_processed[example_index] = []
    id_to_options[example_index] = options_without_pad_before_punctution
    for word in words_unique:
        found = 0
        # iterate through the word and punc pairs, and find the one that matches the word
        for word_and_punc_pair in word_and_punc_pairs:
            # it is a match if pair = word + punc
            for punc in PUNCTUATIONS_LIST:
                if word_and_punc_pair == word + punc:
                    id_to_word_and_punc_pairs_processed[example_index].append(word_and_punc_pair)
                    found = 1
                    break
            if found == 1:
                break
    # calculate the number of correct top num_beams: if the correct word is in the top num_beams, then it is correct
    for word in words_unique:
        if word == data_appended[example_index]['targets_pretokenized'][0]:
            count_correct_top_num_beams += 1
            break
id_to_options_numpy = {}
for key in id_to_options:
    options_numpy = []
    for option in id_to_options[key]:
        options_numpy.append(np.array(option.cpu()))
    id_to_options_numpy[key] = options_numpy

timed_pickle_file_name = '/work/09127/tomyoung/ls6/data/pkls/t5_lambada_vanilla_beam_search_results_' + str(time.time()) + '.pickle'
# Save your data to a pickle file
with open(timed_pickle_file_name, 'wb') as fp:
    pickle.dump({'count_correct': count_correct,
                 'count_correct_top_num_beams': count_correct_top_num_beams,
                 'count_no_words_found': count_no_words_found,
                 'id_to_word_and_punc_pairs': id_to_word_and_punc_pairs,
                 'id_to_word_and_punc_pairs_processed': id_to_word_and_punc_pairs_processed,
                 'id_to_options_numpy': id_to_options_numpy}, fp)


# %%
id_to_options_numpy = {}
for key in id_to_options:
    options_numpy = []
    for option in id_to_options[key]:
        options_numpy.append(np.array(option.cpu()))
    id_to_options_numpy[key] = options_numpy


# %%
len(id_to_options)

# %%
timed_pickle_file_name = '/work/09127/tomyoung/ls6/data/pkls/t5_lambada_vanilla_beam_search_results_' + str(time.time()) + '.pickle'
# Save your data to a pickle file
with open(timed_pickle_file_name, 'wb') as fp:
    pickle.dump({'count_correct': count_correct,
                 'count_correct_top_num_beams': count_correct_top_num_beams,
                 'count_no_words_found': count_no_words_found,
                 'id_to_word_and_punc_pairs': id_to_word_and_punc_pairs,
                 'id_to_word_and_punc_pairs_processed': id_to_word_and_punc_pairs_processed,
                 'id_to_options_numpy': id_to_options_numpy}, fp)

# %%
# load it back
# /work/09127/tomyoung/ls6/inconsistencies_project/t5_lambada_vanilla_beam_search_results_1683476272.4741185.pickle
timed_pickle_file_name = '/work/09127/tomyoung/ls6/data/pkls/t5_lambada_vanilla_beam_search_results_1684795498.9581013.pickle'
with open(timed_pickle_file_name, 'rb') as fp:
    t5_lambada_vanilla_beam_search_results = pickle.load(fp)

# %%
t5_lambada_vanilla_beam_search_results['id_to_options_numpy']

# %%
id_to_options = {}
for key in t5_lambada_vanilla_beam_search_results['id_to_options_numpy']:
    options = []
    for option in t5_lambada_vanilla_beam_search_results['id_to_options_numpy'][key]:
        options.append(torch.from_numpy(option).to(device))
    id_to_options[key] = options

# %%
id_to_options

# %%
input_ids = tokenizer(data_appended[0]['inputs_pretokenized'], return_tensors="pt").input_ids.to(device)
print(data_appended[0]['inputs_pretokenized'])
labels = tokenizer("<extra_id_0> " + id_to_word_and_punc_pairs_processed[0][2] + " <extra_id_1>", return_tensors="pt").input_ids.to(device)
print("<extra_id_0> " + id_to_word_and_punc_pairs_processed[0][2] + " <extra_id_1>")
outputs = model(input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits
logits

# %%
def get_avg_log_p_of_option_without_pad(inputs_pretokenized, option, offset=0):
    # input_ids: 1*len = words + 32099 + 1
    input_ids = tokenizer(inputs_pretokenized, return_tensors="pt").input_ids.to(device)
    # labels: 1*len = 32099 + words
    labels = option.unsqueeze(0).to(device)
    # print('input_ids', input_ids)
    # print('labels', labels)
    # when offset is used, we move the last offset from input_ids to the front of labels.
    if offset != 0:
        to_move = input_ids[0][-offset-2:-2]
        labels = torch.cat((labels[0][0].unsqueeze(0), to_move, labels[0][1:]), dim=0).unsqueeze(0)
        input_ids = torch.cat((input_ids[0][:-offset-2], input_ids[0][-2:]), dim=0).unsqueeze(0)
    # print('input_ids offset', input_ids)
    # print('labels offset', labels)
    outputs = model(input_ids, labels=labels)
    return -outputs.loss, outputs.logits

# %%
def get_offsetted(inputs_pretokenized, option, offset=0):
    # input_ids: 1*len = words + 32099 + 1
    input_ids = tokenizer(inputs_pretokenized, return_tensors="pt").input_ids.to(device)
    # labels: 1*len = 32099 + words
    labels = option.unsqueeze(0).to(device)
    # print('input_ids', input_ids)
    # print('labels', labels)
    # when offset is used, we move the last offset from input_ids to the front of labels.
    if offset != 0:
        to_move = input_ids[0][-offset-2:-2]
        labels = torch.cat((labels[0][0].unsqueeze(0), to_move, labels[0][1:]), dim=0)
        input_ids = torch.cat((input_ids[0][:-offset-2], input_ids[0][-2:]), dim=0)
    else:
        # squeeze the batch dimension
        labels = labels[0]
        input_ids = input_ids[0]
    # print('input_ids offset', input_ids)
    # print('labels offset', labels)
    return (input_ids, labels)

# %%
id_to_options

# save id_and_offset_to_input_and_options to a pickle file
pickle_file_name = '/work/09127/tomyoung/ls6/data/pkls/id_and_offset_to_input_and_options_t5.pickle'
with open(pickle_file_name, 'wb') as fp:
    pickle.dump(id_and_offset_to_input_and_options, fp)

# %%
def get_avg_log_p_of_option_without_pad_batch(inputs_pretokenized_batch, options_batch):
    # input_ids: batch_size*len = words + 32099 + 1
    input_ids = tokenizer(inputs_pretokenized_batch, return_tensors="pt", padding=True).input_ids.to(device)
    labels = options_batch.to(device)
    outputs = model(input_ids, labels=labels)
    return -outputs.loss, outputs.logits

# load it back
# /work/09127/tomyoung/ls6/inconsistencies_project/t5_lambada_vanilla_beam_search_results_1683476272.4741185.pickle
timed_pickle_file_name = '/work/09127/tomyoung/ls6/data/pkls/t5_lambada_vanilla_beam_search_results_1684795498.9581013.pickle'
with open(timed_pickle_file_name, 'rb') as fp:
    t5_lambada_vanilla_beam_search_results = pickle.load(fp)

''' obtain the avg_log_ps '''
import traceback
import datetime
max_offset = 30

id_and_offset_to_option_probs = dict()
failed_example_indices = []
for example_index in tqdm(range(len(data_appended))): # len(data_appended)
    try:
        if len(id_to_options[example_index]) == 0:
            continue
        for offset in range(max_offset):
            options_batch = torch.nn.utils.rnn.pad_sequence([id_and_offset_to_input_and_options[(example_index, offset)][i][1] for i in range(len(id_to_options[example_index]))], batch_first=True, padding_value=tokenizer.pad_token_id)
            input_ids_batch = torch.cat([id_and_offset_to_input_and_options[(example_index, offset)][i][0].unsqueeze(0) for i in range(len(id_to_options[example_index]))], dim=0)
            outputs = model(input_ids_batch, labels=options_batch)
            for option_index in range(len(id_to_options[example_index])):
                avg_log_p = -loss_fn(outputs.logits[option_index][1+offset:], options_batch[option_index][1+offset:]) # [1:] to remove the first token <extra_id_0>
                id_and_offset_to_option_probs[(example_index, offset, option_index)] = avg_log_p.detach().cpu().tolist()
            
            # allocated_memory_bytes = torch.cuda.memory_allocated()
            # # Convert the allocated memory to gigabytes
            # allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
            # print(f"Current GPU memory allocation: {allocated_memory_gb} GB")
    except Exception as e:
        print(f"An error occurred: {e}")
        print('example_index:', example_index, ' failed')
        failed_example_indices.append(example_index)
        traceback.print_exc()

# save avg_log_ps into a pickle file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f'id_and_offset_to_option_probs_{timestamp}.pickle', 'wb') as handle:
    pickle.dump(id_and_offset_to_option_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
failed_example_indices
# save failed_example_indices into a pickle file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f'failed_example_indices_{timestamp}.pickle', 'wb') as handle:
    pickle.dump(failed_example_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# id_and_offset_to_option_probs_20230509-155934.pickle
import pickle
with open('id_and_offset_to_option_probs_20230509-155934.pickle', 'rb') as handle:
    id_and_offset_to_option_probs = pickle.load(handle)
# failed_example_indices_20230509-155934.pickle
with open('failed_example_indices_20230509-155934.pickle', 'rb') as handle:
    failed_example_indices = pickle.load(handle)

'''EOC with max pooling'''
max_offset_test = 10
offset_to_accuracy = dict()

for offset_test in range(max_offset_test):
    count_eoc = 0
    # postprocess the id_and_offset_to_option_probs to get the best option
    for example_index in tqdm(range(len(data_appended))): # len(data_appended)
        if len(id_to_options[example_index]) == 0 or example_index in failed_example_indices:
            continue
        option_avg_log_p_max = -10000000
        best_option =  ""
        for offset in range(offset_test+1):
            for option_index in range(len(id_to_options[example_index])):
                avg_log_p = id_and_offset_to_option_probs[(example_index, offset, option_index)]
                if avg_log_p > option_avg_log_p_max:
                    option_avg_log_p_max = avg_log_p
                    best_option = id_to_options[example_index][option_index]

        best_option_string = tokenizer.decode(best_option)
        # print('best_option_string', best_option_string)
        if get_words_from_options([best_option_string]) != []:
            best_word = get_words_from_options([best_option_string])[0]
            if best_word == data_appended[example_index]['targets_pretokenized'][0]:
                count_eoc += 1
    offset_to_accuracy[offset_test] = count_eoc / (len(data_appended) - len(failed_example_indices))



