from tqdm import tqdm
import copy
import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from argparse import ArgumentParser
import psutil

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python bart_for_generating_testing_data.py --save_file './data/pkls/acceptable_alternatives_ignore_commons_10000.pkl' > nohups/nohup_bart_dec_11.out &

def run():
    parser = ArgumentParser()
    parser.add_argument("--no_stories", type=int, default=10000)
    parser.add_argument("--ignore_common_words", type=bool, default=True)
    # add save file name
    parser.add_argument("--save_file", type=str, default='./data/pkls/acceptable_alternatives.pkl')
    # parser.add_argument("--model_name", type=str, default='t5-3b')
    # parser.add_argument("--cache_dir", type=str, default='./t5-3b-cache')
    
    args = parser.parse_args()


    num_output_sequences = 100
    max_seq_length = 30

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", cache_dir = './facebook-bart-large-cache', forced_bos_token_id=0)
    model = model.cuda()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


    with open('./data/c4-train.00000-of-00512-list-of-lists.json', 'r', encoding='utf8') as f:
        dics_realnewslike = json.load(f)


    top_60_common_words = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', \
        'are', 'as', 'with', 'his', 'they', 'I', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', \
        'but', 'not', 'what', 'all', 'were', 'we', 'whens', 'your', 'can', 'said', 'there', 'use', 'an', 'each', 'she', \
        'which', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so']

    top_60_common_words_cap = ['The', 'Of', 'And', 'A', 'To', 'In', 'Is', 'You', 'That', 'It', 'He', 'Was', 'For', 'On', \
        'Are', 'As', 'With', 'His', 'They', 'I', 'At', 'Be', 'This', 'Have', 'From', 'Or', 'One', 'Had', 'By', 'Word', \
        'But', 'Not', 'What', 'All', 'Were', 'We', 'When' , 'Your', 'Can', 'Said', 'There', 'Use', 'An', 'Each', 'She', \
        'Which', 'Do', 'How', 'Their', 'If', 'Will', 'Up', 'Other', 'About', 'Out', 'Many', 'Then', 'Them', 'These', 'So']

    common_words = top_60_common_words + top_60_common_words_cap


    def differ_only_in_1_or_2_specified_positions(list1, list2, position1, position2):
        assert len(list1) == len(list2)
        num_differences = 0
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                if i == position1 or i == position2:
                    num_differences += 1
                else:
                    return False
        return True

    acceptable_alternatives = dict()

        
    for story_id in tqdm(range(args.no_stories)):
        for paragraph_id in range(len(dics_realnewslike[story_id])):
            # get the raw sequence
            example_raw_sequence = dics_realnewslike[story_id][paragraph_id]
            # tokenize the raw sequence
            example_raw_sequence_bart_tokenized = tokenizer.tokenize(example_raw_sequence)
            if len(example_raw_sequence_bart_tokenized) > max_seq_length:
                continue
            for i in range(len(example_raw_sequence_bart_tokenized) - 1):
                # it has to start with 'Ġ'
                if example_raw_sequence_bart_tokenized[i][0] != 'Ġ':
                    continue
                # isalpha()
                if not example_raw_sequence_bart_tokenized[i][1:].isalpha():
                    continue
                # the next character has to start with 'Ġ'
                if example_raw_sequence_bart_tokenized[i+1][0] != 'Ġ':
                    continue
                # isalpha()
                if not example_raw_sequence_bart_tokenized[i+1][1:].isalpha():
                    continue
                # if the next next character exists, it has to start with 'Ġ'
                if len(example_raw_sequence_bart_tokenized) > i+2:
                    if example_raw_sequence_bart_tokenized[i+2][0] != 'Ġ':
                        continue
                    # isalpha()
                    if not example_raw_sequence_bart_tokenized[i+2][1:].isalpha():
                        continue   
                
                example_raw_sequence_bart_tokenized_copy = copy.deepcopy(example_raw_sequence_bart_tokenized)
                input_ids_original = tokenizer.convert_tokens_to_ids(example_raw_sequence_bart_tokenized_copy)
                # add bos and eos token ids
                input_ids_original = [tokenizer.bos_token_id] + input_ids_original + [tokenizer.eos_token_id]
                input_ids_original = torch.tensor(input_ids_original).unsqueeze(0)

                # for each token in the sequence, replace it with the mask token
                example_raw_sequence_bart_tokenized_masked = copy.deepcopy(example_raw_sequence_bart_tokenized)
                example_raw_sequence_bart_tokenized_masked[i:i+2] = ['<mask>']
                # convert the tokenized sequence to input_ids
                input_ids = tokenizer.convert_tokens_to_ids(example_raw_sequence_bart_tokenized_masked)
                # add bos and eos token ids
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                # convert input_ids to a tensor'
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                # generate the output sequences
                # catch torch.cuda.OutOfMemoryError
                try:
                    outputs = model.generate(input_ids.to('cuda'),
                                                return_dict_in_generate=True,
                                                num_beams=num_output_sequences,
                                                output_scores=True,
                                                num_return_sequences=num_output_sequences,
                                                max_new_tokens=1000)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print((story_id, paragraph_id, i))
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                
                # lose all <pad>
                sequences = [seq[seq != tokenizer.pad_token_id] for seq in outputs.sequences]

                # for the output sequence to be acceptable, two conditions must be met:
                # 1. the output sequence must be the same length as the input sequence (plus 1 for the bos token)
                # 2. the output sequence must be only different from the input sequence at the masked token position
                acceptable_sequences = []
                new_bigrams = []
                for j in range(num_output_sequences):
                    if len(sequences[j]) == len(input_ids[0]) + 2:
                        if differ_only_in_1_or_2_specified_positions(sequences[j][1:].tolist(), input_ids_original[0].tolist(), i+1, i+2):
                            acceptable_sequences.append(sequences[j][1:].tolist())
                            new_bigrams.append(sequences[j][i+2:i+4].tolist())

                # add to the dictionary
                if len(acceptable_sequences) > 0:
                    acceptable_alternatives[(story_id, paragraph_id, i)] = \
                        (acceptable_sequences,
                            new_bigrams,
                            [tokenizer.decode(seq, skip_special_tokens=False) for seq in acceptable_sequences],
                            [tokenizer.decode(bigram, skip_special_tokens=False) for bigram in new_bigrams])

        # save the dictionary as a pickle file
        import pickle
        with open(args.save_file, 'wb') as f:
            pickle.dump(acceptable_alternatives, f)

if __name__ == '__main__':
    run()