import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer
import torch
import copy
from argparse import ArgumentParser
from tqdm import tqdm
import pickle

# nohup srun -p gpu-a100 -n 1 -t 20:00:00 python calculate_inconsistencies_t5_c4.py \
# --url_to_probs_dict_file './data/pkls/url_to_probs_c4_dict_with_labels_t5_11b.pkl' \
# --c4_json_file './data/c4-train.00000-of-00512-list-of-lists.json' \
# --model_name 't5-11b' \
# --model_parallelism \
# --cache_dir './t5-11b-cache' > nohups/nohup_t5_11b_c4_labels.out &


def run():
    parser = ArgumentParser()
    parser.add_argument("--no_samples", type=int, default=10000)
    parser.add_argument("--model_name", type=str, default='t5-3b')
    parser.add_argument("--si2bsp2op_file", type=str, default='si2bsp2op-t5-3b-3000.json')
    parser.add_argument("--acceptable_alternatives_file", type=str, default='./data/pkls/acceptable_alternatives.pkl')
    parser.add_argument("--cache_dir", type=str, default='./t5-3b-cache')
    parser.add_argument("--url_to_probs_dict_file", type=str, default='./data/pkls/url_to_probs_dict.pkl')
    parser.add_argument("--c4_json_file", type=str, default='c4-train.00000-of-00512-list-of-lists.json')
    parser.add_argument('--model_parallelism', action='store_true')
    parser.add_argument('--no-model_parallelism', dest='ignore_common_words', action='store_false')
    parser.set_defaults(model_parallelism=True)
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    extra_id_0_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    extra_id_1_id = tokenizer.convert_tokens_to_ids('<extra_id_1>')

    with open(args.c4_json_file, 'r', encoding='utf8') as f:
        dicts_realnewslike = json.load(f)


    # load acceptable_alternatives from a pkl file
    with open(args.acceptable_alternatives_file, 'rb') as f:
        acceptable_alternatives = pickle.load(f)


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

    # remove instances where only 1 option is given (which is usually the original sequence)
    acceptable_alternatives_multi = dict()
    for key in acceptable_alternatives:
        if len(acceptable_alternatives[key][0]) > 1:
            acceptable_alternatives_multi[key] = acceptable_alternatives[key]


    paired_keys = []
    # we want to find the keys that have the same story_id and paragraph_id and their positions differ by 1
    for k1 in acceptable_alternatives_multi:
        for k2 in acceptable_alternatives_multi:
            if k1[0] == k2[0] and k1[1] == k2[1] and abs(k1[2] - k2[2])==1:
                paired_keys.append((k1, k2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_parallelism:
        model.parallelize()
    else:
        model.to(device)

    url_to_probs_dict = dict()
    for pair in tqdm(paired_keys):
        # pair[0] is the new proposed sequence
        for option_id in range(len(acceptable_alternatives_multi[pair[0]][0])):
            # if the option is the same as the original sequence, skip
            if acceptable_alternatives_multi[pair[0]][3][option_id] == '<s>' + dicts_realnewslike[pair[0][0]][pair[0][1]] + '</s>':
                continue
            # confirm that in the eyes of T5, the new sequence is indeed only different by 1 token from the original sequence
            T5_tokenized_original = tokenizer.tokenize(dicts_realnewslike[pair[0][0]][pair[0][1]])
            T5_tokenized_proposed = tokenizer.tokenize(acceptable_alternatives_multi[pair[0]][3][option_id].replace('<s>', '').replace('</s>', ''))
            
            if not (len(T5_tokenized_original) == len(T5_tokenized_proposed) and \
            differ_only_in_1_position(T5_tokenized_original, T5_tokenized_proposed)):
                # print('something wrong')
                # print(dicts_realnewslike[pair[0][0]][pair[0][1]])
                # print(acceptable_alternatives_multi[pair[0]][3][option_id].replace('<s>', '').replace('</s>', ''))
                continue

            # make sure that the T5 tokenizer also see both the proposed and original words as one token
            proposed_word = bart_tokenizer.convert_ids_to_tokens(acceptable_alternatives_multi[pair[0]][1][option_id]).replace('Ġ', '')
            proposed_word_tokenized_t5_style_expected = '▁' + proposed_word
            if proposed_word_tokenized_t5_style_expected not in T5_tokenized_proposed:
                print('T5 tokenizer sees the proposed word as more than one token')
                print(proposed_word_tokenized_t5_style_expected)
                print(T5_tokenized_proposed)
                continue
            
            bart_tokenized_original = bart_tokenizer.tokenize(dicts_realnewslike[pair[0][0]][pair[0][1]])
            original_word = bart_tokenized_original[pair[0][2]].replace('Ġ', '')
            original_word_tokenized_t5_style_expected = '▁' + original_word
            if original_word_tokenized_t5_style_expected not in T5_tokenized_original:
                print('T5 tokenizer sees the original word as more than one token')
                print(original_word_tokenized_t5_style_expected)
                print(T5_tokenized_original)
                continue

            # check pair[1][2] to learn if we want to go left or right
            go_left = True
            if pair[1][2] - pair[0][2] == 1:
                go_left = False
            
            # the position of interest is where T5_tokenized_original is different from T5_tokenized_proposed
            position_of_interest = -1
            for i in range(len(T5_tokenized_original)):
                if T5_tokenized_original[i] != T5_tokenized_proposed[i]:
                    position_of_interest = i
                    break
            # the other position has to be a standalone word
            if go_left:
                the_other_position = position_of_interest - 1
                if T5_tokenized_original[the_other_position][0] != '▁':
                    print('the other position is not a standalone word')
                    print(T5_tokenized_original)
                    print(T5_tokenized_original[the_other_position])
                    continue
            else:
                the_other_position = position_of_interest + 1
                if T5_tokenized_original[the_other_position][0] != '▁':
                    print('the other position is not a standalone word')
                    print(T5_tokenized_original)
                    print(T5_tokenized_original[the_other_position])
                    continue

            # compute the joint probs of the original bigram and the proposed bigram 
            T5_tokenized_original_temp = copy.deepcopy(T5_tokenized_original)
            T5_tokenized_proposed_temp = copy.deepcopy(T5_tokenized_proposed)


            url_to_probs_dict[(pair[0], pair[1], option_id)] = dict()
            # mask the original bigram with <extra_id_0>
            if not go_left:
                original_bigram = T5_tokenized_original_temp[position_of_interest:position_of_interest+2]
                proposed_bigram = T5_tokenized_proposed_temp[position_of_interest:position_of_interest+2]
                T5_tokenized_original_temp[position_of_interest:position_of_interest+2] = ['<extra_id_0>']
            else:
                original_bigram = T5_tokenized_original_temp[position_of_interest-1:position_of_interest+1]
                proposed_bigram = T5_tokenized_proposed_temp[position_of_interest-1:position_of_interest+1]
                T5_tokenized_original_temp[position_of_interest-1:position_of_interest+1] = ['<extra_id_0>']


            input_ids = tokenizer.convert_tokens_to_ids(T5_tokenized_original_temp) + [tokenizer.eos_token_id]
            labels_original_bigram = [extra_id_0_id] + \
                tokenizer.convert_tokens_to_ids(original_bigram) + \
                [extra_id_1_id] + \
                [tokenizer.eos_token_id]
            print('the joint probability of :' + tokenizer.convert_tokens_to_string(original_bigram))
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            print(tokenizer.convert_ids_to_tokens(labels_original_bigram))

            outputs = model(input_ids = torch.tensor([input_ids]).to(device), \
                            labels = torch.tensor([labels_original_bigram]).to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_to_save = []
            for i in range(1, len(labels_original_bigram)-1):
                probs_to_save.append(probs[0][i][labels_original_bigram[i]].tolist())
            url_to_probs_dict[(pair[0], pair[1], option_id)][tuple(tokenizer.convert_tokens_to_ids(original_bigram))] = probs_to_save
            url_to_probs_dict[(pair[0], pair[1], option_id)]['original bigram'] = tuple(tokenizer.convert_tokens_to_ids(original_bigram))
            print('probs:')
            print(probs_to_save)


            labels_proposed_bigram = [extra_id_0_id] + \
                tokenizer.convert_tokens_to_ids(proposed_bigram) + \
                [extra_id_1_id] + \
                [tokenizer.eos_token_id]
            print('the joint probability of :' + tokenizer.convert_tokens_to_string(proposed_bigram))
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            print(tokenizer.convert_ids_to_tokens(labels_proposed_bigram))

            outputs = model(input_ids = torch.tensor([input_ids]).to(device), \
                            labels = torch.tensor([labels_proposed_bigram]).to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_to_save = []
            for i in range(1, len(labels_proposed_bigram)-1):
                probs_to_save.append(probs[0][i][labels_proposed_bigram[i]].tolist())
            url_to_probs_dict[(pair[0], pair[1], option_id)][tuple(tokenizer.convert_tokens_to_ids(proposed_bigram))] = probs_to_save
            url_to_probs_dict[(pair[0], pair[1], option_id)]['proposed bigram'] = tuple(tokenizer.convert_tokens_to_ids(proposed_bigram))
            print('probs:')
            print(probs_to_save)

            # compute the conditional probs of the original token and the proposed token
            T5_tokenized_original_temp = copy.deepcopy(T5_tokenized_original)
            T5_tokenized_proposed_temp = copy.deepcopy(T5_tokenized_proposed)

            T5_tokenized_original_temp[position_of_interest] = '<extra_id_0>'
            input_ids = tokenizer.convert_tokens_to_ids(T5_tokenized_original_temp) + [tokenizer.eos_token_id]
            labels_original_token = [extra_id_0_id] + \
                [tokenizer.convert_tokens_to_ids(T5_tokenized_original[position_of_interest])] + \
                [extra_id_1_id] + \
                [tokenizer.eos_token_id]
            print('the conditional probability of :' + T5_tokenized_original[position_of_interest])
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            print(tokenizer.convert_ids_to_tokens(labels_original_token))

            outputs = model(input_ids = torch.tensor([input_ids]).to(device), \
                            labels = torch.tensor([labels_original_token]).to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_to_save = []
            for i in range(1, len(labels_original_token)-1):
                probs_to_save.append(probs[0][i][labels_original_token[i]].tolist())
            url_to_probs_dict[(pair[0], pair[1], option_id)][tokenizer.convert_tokens_to_ids(T5_tokenized_original[position_of_interest])] = probs_to_save
            url_to_probs_dict[(pair[0], pair[1], option_id)]['original token'] = tokenizer.convert_tokens_to_ids(T5_tokenized_original[position_of_interest])


            print('probs:')
            print(probs_to_save)
            labels_proposed_token = [extra_id_0_id] + \
                [tokenizer.convert_tokens_to_ids(T5_tokenized_proposed[position_of_interest])] + \
                [extra_id_1_id] + \
                [tokenizer.eos_token_id]
            print('the conditional probability of :' + T5_tokenized_proposed[position_of_interest])
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            print(tokenizer.convert_ids_to_tokens(labels_proposed_token))

            outputs = model(input_ids = torch.tensor([input_ids]).to(device), \
                            labels = torch.tensor([labels_proposed_token]).to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            probs_to_save = []
            for i in range(1, len(labels_proposed_token)-1):
                probs_to_save.append(probs[0][i][labels_proposed_token[i]].tolist())
            url_to_probs_dict[(pair[0], pair[1], option_id)][tokenizer.convert_tokens_to_ids(T5_tokenized_proposed[position_of_interest])] = probs_to_save
            url_to_probs_dict[(pair[0], pair[1], option_id)]['proposed token'] = tokenizer.convert_tokens_to_ids(T5_tokenized_proposed[position_of_interest])

            print('probs:')
            print(probs_to_save)
            
            

    with open(args.url_to_probs_dict_file, 'wb') as f:
        pickle.dump(url_to_probs_dict, f)


if __name__ == '__main__':
    run()