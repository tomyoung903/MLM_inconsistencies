from tqdm import tqdm
import copy
import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from argparse import ArgumentParser
from transformers.generation_utils import BeamSearchEncoderDecoderOutput as BSOutput
import pickle
import psutil
import os

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python bart_for_generating_testing_data_parallel.py --no_stories 100 --save_file './data/pkls/acceptable_alternatives_ignore_commons_10000.pkl' > nohups/nohup_bart_dec_12.out &

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python bart_for_generating_testing_data_parallel.py --no_stories 10000 \
# --input_ids_cache './data/pkls/input_ids_cache_10000_ignore_cws.pkl' \
# --url_tuples_cache './data/pkls/url_tuples_cache_10000_ignore_cws.pkl' \
#  --no-ignore_common_words --save_file './data/pkls/acceptable_alternatives_10000_ignore_cws.pkl' > nohups/nohup_bart_dec_12.out &

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python3 bart_for_generating_testing_data_parallel.py --no_stories 1000 --input_ids_cache './data/pkls/input_ids_cache_1000_ignore_cws_nos_50_valid.pkl' --url_tuples_cache './data/pkls/url_tuples_cache_1000_ignore_cws_nos_50_valid.pkl' --num_output_sequences 50 --batch_size 1  --ignore_common_words --save_file './data/pkls/acceptable_alternatives_1000_ignore_cws_nos_50_valid.pkl' --print_memory_usage 'no' --corpus_file './data/c4-validation.00000-of-00001-list-of-lists.json' > nohups/nohup_bart_nos_50_valid.out &

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python bart_for_generating_testing_data_parallel.py --no_stories 1000 \
# --input_ids_cache './data/pkls/input_ids_cache_1000_ignore_cws_nos_100.pkl' \
# --url_tuples_cache './data/pkls/url_tuples_cache_1000_ignore_cws_nos_100.pkl' \
# --num_output_sequences 100 \
#  --no-ignore_common_words --save_file './data/pkls/acceptable_alternatives_1000_ignore_cws.pkl' > nohups/nohup_bart_nos_100.out &


def run():
    parser = ArgumentParser()
    parser.add_argument("--no_stories", type=int, default=10000)
    parser.add_argument('--ignore_common_words', action='store_true')
    parser.add_argument('--no-ignore_common_words', dest='ignore_common_words', action='store_false')
    parser.set_defaults(ignore_common_words=True)
    # add save file name
    parser.add_argument("--save_file", type=str, default='./data/pkls/acceptable_alternatives.pkl')
    parser.add_argument("--batch_size", type=int, default=32)
    # add input_ids cache
    parser.add_argument("--input_ids_cache", type=str, default='./data/pkls/input_ids_cache.pkl')
    # add url tuples cache
    parser.add_argument("--url_tuples_cache", type=str, default='./data/pkls/url_tuples_cache.pkl')
    parser.add_argument("--num_output_sequences", type=int, default=10)
    parser.add_argument("--print_memory_usage", type=str, default='no')
    # add corpus file
    parser.add_argument("--corpus_file", type=str, default='./data/c4-train.00000-of-00512-list-of-lists.json')
    
    # parser.add_argument("--model_name", type=str, default='t5-3b')
    # parser.add_argument("--cache_dir", type=str, default='./t5-3b-cache')
    
    args = parser.parse_args()
    print(args)

    num_output_sequences = args.num_output_sequences
    max_seq_length = 30

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", cache_dir = './facebook-bart-large-cache', forced_bos_token_id=0)
    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


    with open(args.corpus_file, 'r', encoding='utf8') as f:
        dicts_realnewslike = json.load(f)


    top_60_common_words = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', \
        'are', 'as', 'with', 'his', 'they', 'I', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', \
        'but', 'not', 'what', 'all', 'were', 'we', 'whens', 'your', 'can', 'said', 'there', 'use', 'an', 'each', 'she', \
        'which', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so']

    top_60_common_words_cap = ['The', 'Of', 'And', 'A', 'To', 'In', 'Is', 'You', 'That', 'It', 'He', 'Was', 'For', 'On', \
        'Are', 'As', 'With', 'His', 'They', 'I', 'At', 'Be', 'This', 'Have', 'From', 'Or', 'One', 'Had', 'By', 'Word', \
        'But', 'Not', 'What', 'All', 'Were', 'We', 'When' , 'Your', 'Can', 'Said', 'There', 'Use', 'An', 'Each', 'She', \
        'Which', 'Do', 'How', 'Their', 'If', 'Will', 'Up', 'Other', 'About', 'Out', 'Many', 'Then', 'Them', 'These', 'So']

    common_words = top_60_common_words + top_60_common_words_cap


    def differ_only_in_1_position(list1, list2):
        assert len(list1) == len(list2)
        num_differences = 0
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                num_differences += 1
        if num_differences == 1:
            return True
        else:
            return False

    seq_length_after_padding = max_seq_length + 5

    url_tuples = []
    input_ids_all = None
    # if the input_ids cache and url tuples cache exists, load it
    if os.path.exists(args.input_ids_cache) and os.path.exists(args.url_tuples_cache):
        with open(args.input_ids_cache, 'rb') as f:
            input_ids_all = pickle.load(f)
        with open(args.url_tuples_cache, 'rb') as f:
            url_tuples = pickle.load(f)
    else:
        for story_id in tqdm(range(args.no_stories)):
            for paragraph_id in range(len(dicts_realnewslike[story_id])):
                # get the raw sequence
                example_raw_sequence = dicts_realnewslike[story_id][paragraph_id]
                # tokenize the raw sequencecv
                example_raw_sequence_bart_tokenized = tokenizer.tokenize(example_raw_sequence)
                if len(example_raw_sequence_bart_tokenized) > max_seq_length:
                    continue
                for i in range(len(example_raw_sequence_bart_tokenized)):
                    # it has to start with 'Ġ'
                    if example_raw_sequence_bart_tokenized[i][0] != 'Ġ':
                        continue
                    # the next character has to start with 'Ġ'
                    if i < len(example_raw_sequence_bart_tokenized) - 1:
                        if example_raw_sequence_bart_tokenized[i+1][0] != 'Ġ':
                            continue
                    # isalpha()
                    if not example_raw_sequence_bart_tokenized[i][1:].isalpha():
                        continue
                    # if the word is a common word, skip it
                    if args.ignore_common_words:
                        if example_raw_sequence_bart_tokenized[i][1:] in common_words:
                            continue

                    # for each token in the sequence, replace it with the mask token
                    example_raw_sequence_bart_tokenized_masked = copy.deepcopy(example_raw_sequence_bart_tokenized)
                    example_raw_sequence_bart_tokenized_masked[i] = '<mask>'
                    # convert the tokenized sequence to input_ids
                    input_ids = tokenizer.convert_tokens_to_ids(example_raw_sequence_bart_tokenized_masked)
                    # add bos and eos token ids
                    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                    # convert input_ids to a tensor'
                    input_ids = torch.tensor(input_ids).unsqueeze(0)
                    # pad the input_ids
                    input_ids = torch.nn.functional.pad(input_ids, (0, seq_length_after_padding - len(input_ids[0])), value=tokenizer.pad_token_id)
                    # add to the batch
                    if input_ids_all is None:
                        input_ids_all = input_ids
                    else:
                        input_ids_all = torch.cat((input_ids_all, input_ids), dim=0)
                    # add to the url_tuples
                    url_tuples.append((story_id, paragraph_id, i))
        # save the input_ids and url_tuples
        with open(args.input_ids_cache, 'wb') as f:
            pickle.dump(input_ids_all, f)
        with open(args.url_tuples_cache, 'wb') as f:
            pickle.dump(url_tuples, f)
        
    if torch.cuda.is_available():
        input_ids_all = input_ids_all.cuda()
    
    batch_size = args.batch_size
    num_batches = len(url_tuples) // batch_size
    acceptable_alternatives_parallel_batched = dict()
    for batch_id in tqdm(range(num_batches)):
        # generate the output sequences
        outputs_batch = model.generate(input_ids_all[batch_id*batch_size:(batch_id+1)*batch_size],
                                    return_dict_in_generate=True,
                                    num_beams=num_output_sequences,
                                    output_scores=True,
                                    num_return_sequences=num_output_sequences,
                                    max_new_tokens=1000)
        # split the outputs_batch into a list of outputs
        outputs_list = []
        for i in range(batch_size):
            new_outputs = BSOutput()
            new_outputs.sequences = outputs_batch.sequences[i*num_output_sequences:(i+1)*num_output_sequences]
            # new_outputs.scores = tuple()
            # for j in range(len(outputs_batch.scores)):
            #     new_outputs.scores += (outputs_batch.scores[j][i*num_output_sequences:(i+1)*num_output_sequences],)
            new_outputs.beam_indices = outputs_batch.beam_indices[i*num_output_sequences:(i+1)*num_output_sequences]
            outputs_list.append(new_outputs)
        for i in range(batch_size):
            position = url_tuples[i + batch_id * batch_size][2]
            input_ids = input_ids_all[i + batch_id * batch_size]
            current_outputs = outputs_list[i]
            # lose all <pad> from input_ids
            input_ids = input_ids[input_ids != tokenizer.pad_token_id]
            # unsqueeze to make it a batch of size 1
            input_ids = input_ids.unsqueeze(0)

            # lose all <pad>
            sequences = [seq[seq != tokenizer.pad_token_id] for seq in outputs_list[i].sequences]

            # for the output sequence to be acceptable, two conditions must be met:
            # 1. the output sequence must be the same length as the input sequence (plus 1 for the bos token)
            # 2. the output sequence must be only different from the input sequence at the masked token position
            acceptable_sequences = []
            probs_of_new_token = []
            new_tokens = []
            for j in range(num_output_sequences):
                if len(sequences[j]) == len(input_ids[0]) + 1:
                    if differ_only_in_1_position(sequences[j][1:].tolist(), input_ids[0].tolist()):
                        acceptable_sequences.append(sequences[j][1:].tolist())
                        new_tokens.append(sequences[j][position+2].tolist())
                        probs_of_new_token.append(torch.exp(outputs_batch.scores[position+1][current_outputs.beam_indices[j][position+1]][current_outputs.sequences[j][position+2]]).tolist())

            # add to the dictionary
            if len(acceptable_sequences) > 0:
                acceptable_alternatives_parallel_batched[(url_tuples[i + batch_id * batch_size])] = \
                    (acceptable_sequences,
                        new_tokens,
                        probs_of_new_token,
                        [tokenizer.decode(seq, skip_special_tokens=False) for seq in acceptable_sequences])    
            
                # Get the current process
                # process = psutil.Process()
                # Print the memory usage of the current process in bytes
                # print(process.memory_info().rss)
        # print all variables on device cuda:0
        if torch.cuda.is_available() and args.print_memory_usage == 'yes':
            print('torch.cuda.memory_stats()')
            print(torch.cuda.memory_stats())
            print('torch.cuda.memory_summary()')
            print(torch.cuda.memory_summary())

            # print current cuda memory usage
            print('torch.cuda.memory_allocated()')
            print(torch.cuda.memory_allocated())
            print('torch.cuda.memory_cached()')
            print(torch.cuda.memory_cached())
            print('torch.cuda.memory_reserved()')
            print(torch.cuda.memory_reserved())
            # free up memory
            # del outputs_batch
            # del outputs_list
            # del input_ids
            # del current_outputs
            # del acceptable_sequences
            # del probs_of_new_token
            # del new_tokens
            torch.cuda.empty_cache()

        if batch_id % int(num_batches / 10) == 0:
            print('batch_id = ' + str(batch_id) + ' out of ' + str(num_batches))
            # save the dictionary as a pickle file
            with open(args.save_file, 'wb') as f:
                pickle.dump(acceptable_alternatives_parallel_batched, f)

if __name__ == '__main__':
    run()