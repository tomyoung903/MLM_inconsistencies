""" This script is used to find the inconsistencies in the conditional probabilities of bigrams in the four sentences. This is the "unnormalized" version. 
'unnormalized' means that the probabilities are not normalized. 
See line 183 /work/09127/tomyoung/ls6/inconsistencies_project/possibly_deprecated/roberta_for_bigram_inconsistencies_in_bulk.py for the normalized (incorrect) version."""

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import copy
import sympy as sym
import numpy as np
import pickle
from argparse import ArgumentParser
import math

from scipy.special import rel_entr
from tqdm import tqdm


def run():
    parser = ArgumentParser()

    parser.add_argument("--no_samples", type=int, default=10000)
    parser.add_argument("--model_name", type=str, default='roberta-base')
    # add cache dir
    parser.add_argument("--cache_dir", type=str, default='./roberta-base-cache')
    parser.add_argument("--analytics_filename", type=str, default='roberta-base-on-bigrams.pkl')
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(args.model_name, cache_dir = args.cache_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Open the pickle file and load the data
    with open('./data/pkls/acceptable_alternatives_bigrams.pkl', 'rb') as f:
        data = pickle.load(f)

    # find those with four connected bigrams
    acceptable_alternatives = data
    for key in acceptable_alternatives:
        bigrams = acceptable_alternatives[key][1]
        four_connected_bigrams = (0, 0, 0, 0)
        for i in range(len(bigrams)):
            for j in range(i+1, len(bigrams)):
                if bigrams[i][0] == bigrams[j][0] or bigrams[i][1] == bigrams[j][1]:
                    continue
                if [bigrams[i][0], bigrams[j][1]] in bigrams and \
                    [bigrams[j][0], bigrams[i][1]] in bigrams:
                    four_connected_bigrams = (bigrams[i], bigrams[j], [bigrams[i][0], bigrams[j][1]], [bigrams[j][0], bigrams[i][1]])
                    acceptable_alternatives[key] = acceptable_alternatives[key] + (four_connected_bigrams,)
                    print(key, four_connected_bigrams)
                    print(acceptable_alternatives[key][2][i])
                    print(acceptable_alternatives[key][2][j])
                    print(tokenizer.convert_ids_to_tokens(four_connected_bigrams[2]))
                    print(tokenizer.convert_ids_to_tokens(four_connected_bigrams[3]))
                    break
            if four_connected_bigrams != (0, 0, 0, 0):
                break
    acceptable_alternatives_with_four_connected_bigrams = dict()
    for key in acceptable_alternatives:
        if len(acceptable_alternatives[key]) == 5:
            acceptable_alternatives_with_four_connected_bigrams[key] = acceptable_alternatives[key]

    urls = list(acceptable_alternatives_with_four_connected_bigrams.keys())
    analytics = dict()
    for url_id in tqdm(range(len(urls))):
        url = urls[url_id]
        four_bigrams = acceptable_alternatives_with_four_connected_bigrams[url][4]
        four_sentences = []
        for bigram in four_bigrams:
            # find its index in acceptable_alternatives_with_four_connected_bigrams[(1,0,9)][1]
            index = acceptable_alternatives_with_four_connected_bigrams[url][1].index(bigram)
            # find the corresponding sentence in acceptable_alternatives_with_four_connected_bigrams[(1,0,9)][2]
            sentence = acceptable_alternatives_with_four_connected_bigrams[url][2][index]
            four_sentences.append(sentence)
        four_sentences_without_delimiters = []
        for sentence in four_sentences:
            # lose the <s> from the start of the sentence
            if sentence.startswith('<s>'):
                sentence = sentence[3:]
            # lose the </s> from the end of the sentence
            if sentence.endswith('</s>'):
                sentence = sentence[:-4]
            four_sentences_without_delimiters.append(sentence)
        four_sentences_bert_tokenized = []
        for sentence in four_sentences_without_delimiters:
            four_sentences_bert_tokenized.append(tokenizer.tokenize(sentence))

        if not len(four_sentences_bert_tokenized[0]) == len(four_sentences_bert_tokenized[1]) == len(four_sentences_bert_tokenized[2]) == len(four_sentences_bert_tokenized[3]):
            print('not equal length')
            continue
        

        # there can only be two indices where the four sentences differ
        count = 0
        indices_where_sentences_differ = []
        for i in range(len(four_sentences_bert_tokenized[0])):
            if four_sentences_bert_tokenized[0][i] != four_sentences_bert_tokenized[1][i] or \
                four_sentences_bert_tokenized[0][i] != four_sentences_bert_tokenized[2][i] or \
                four_sentences_bert_tokenized[0][i] != four_sentences_bert_tokenized[3][i]:
                count += 1
                indices_where_sentences_differ.append(i)
        if count != 2:
            print('not exactly two indices differ')
            continue

        # assert that in those indices, the four bigrams are in the right pattern
        vocabulary_0th_index = list(set([four_sentences_bert_tokenized[0][indices_where_sentences_differ[0]],\
            four_sentences_bert_tokenized[1][indices_where_sentences_differ[0]],\
                four_sentences_bert_tokenized[2][indices_where_sentences_differ[0]],\
                    four_sentences_bert_tokenized[3][indices_where_sentences_differ[0]]]))
        if len(vocabulary_0th_index) != 2:
            print('not exactly two words in the first index')
            continue

        vocabulary_1st_index = list(set([four_sentences_bert_tokenized[0][indices_where_sentences_differ[1]],\
            four_sentences_bert_tokenized[1][indices_where_sentences_differ[1]],\
                four_sentences_bert_tokenized[2][indices_where_sentences_differ[1]],\
                    four_sentences_bert_tokenized[3][indices_where_sentences_differ[1]]]))
        if len(vocabulary_1st_index) != 2:
            print('not exactly two words in the second index')
            continue

        word_00 = vocabulary_0th_index[0]
        word_00_id = tokenizer.convert_tokens_to_ids(word_00)
        word_01 = vocabulary_0th_index[1]
        word_01_id = tokenizer.convert_tokens_to_ids(word_01)
        word_10 = vocabulary_1st_index[0]
        word_10_id = tokenizer.convert_tokens_to_ids(word_10)
        word_11 = vocabulary_1st_index[1]
        word_11_id = tokenizer.convert_tokens_to_ids(word_11)

        template_input = tokenizer(four_sentences_without_delimiters[0], return_tensors='pt')

        # obtain conditionals given word_00
        template_input_masked = copy.deepcopy(template_input)
        #                                                          this accounts for the added delimiter
        template_input_masked['input_ids'][0][indices_where_sentences_differ[0] + 1] = tokenizer.convert_tokens_to_ids(vocabulary_0th_index[0])
        template_input_masked['input_ids'][0][indices_where_sentences_differ[1] + 1] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(**template_input_masked.to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        # obtain the conditional of word_10
        conditional_word_10_given_word_00 = probs[0][indices_where_sentences_differ[1] + 1][word_10_id].tolist()
        # obtain the conditional of word_11
        conditional_word_11_given_word_00 = probs[0][indices_where_sentences_differ[1] + 1][word_11_id].tolist()


        # obtain conditionals given word_01
        template_input_masked = copy.deepcopy(template_input)
        #                                                          this accounts for the added delimiter
        template_input_masked['input_ids'][0][indices_where_sentences_differ[0] + 1] = tokenizer.convert_tokens_to_ids(vocabulary_0th_index[1])
        template_input_masked['input_ids'][0][indices_where_sentences_differ[1] + 1] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(**template_input_masked.to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        # obtain the conditional of word_10
        conditional_word_10_given_word_01 = probs[0][indices_where_sentences_differ[1] + 1][word_10_id].tolist()
        # obtain the conditional of word_11
        conditional_word_11_given_word_01 = probs[0][indices_where_sentences_differ[1] + 1][word_11_id].tolist()


        # obtain the conditional given word_10
        template_input_masked = copy.deepcopy(template_input)
        #                                                          this accounts for the added delimiter
        template_input_masked['input_ids'][0][indices_where_sentences_differ[0] + 1] = tokenizer.mask_token_id
        template_input_masked['input_ids'][0][indices_where_sentences_differ[1] + 1] = tokenizer.convert_tokens_to_ids(vocabulary_1st_index[0])
        with torch.no_grad():
            logits = model(**template_input_masked.to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        # obtain the conditional of word_00
        conditional_word_00_given_word_10 = probs[0][indices_where_sentences_differ[0] + 1][word_00_id].tolist()
        # obtain the conditional of word_01
        conditional_word_01_given_word_10 = probs[0][indices_where_sentences_differ[0] + 1][word_01_id].tolist()


        # obtain the conditional given word_11
        template_input_masked = copy.deepcopy(template_input)
        #                                                          this accounts for the added delimiter
        template_input_masked['input_ids'][0][indices_where_sentences_differ[0] + 1] = tokenizer.mask_token_id
        template_input_masked['input_ids'][0][indices_where_sentences_differ[1] + 1] = tokenizer.convert_tokens_to_ids(vocabulary_1st_index[1])
        with torch.no_grad():
            logits = model(**template_input_masked.to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        # obtain the conditional of word_00
        conditional_word_00_given_word_11 = probs[0][indices_where_sentences_differ[0] + 1][word_00_id].tolist()
        # obtain the conditional of word_01
        conditional_word_01_given_word_11 = probs[0][indices_where_sentences_differ[0] + 1][word_01_id].tolist()

        # normalize the conditionals
        conditional_word_00_given_word_10_symbol = sym.symbols('conditional_word_00_given_word_10')
        conditional_word_01_given_word_10_symbol = sym.symbols('conditional_word_01_given_word_10')
        conditional_word_00_given_word_11_symbol = sym.symbols('conditional_word_00_given_word_11')
        conditional_word_01_given_word_11_symbol = sym.symbols('conditional_word_01_given_word_11')
        conditional_word_10_given_word_00_symbol = sym.symbols('conditional_word_10_given_word_00')
        conditional_word_11_given_word_00_symbol = sym.symbols('conditional_word_11_given_word_00')
        conditional_word_10_given_word_01_symbol = sym.symbols('conditional_word_10_given_word_01')
        conditional_word_11_given_word_01_symbol = sym.symbols('conditional_word_11_given_word_01')

        lhs = [(conditional_word_10_given_word_00_symbol / conditional_word_11_given_word_00_symbol) * \
        (conditional_word_00_given_word_11_symbol / conditional_word_01_given_word_11_symbol)]

        rhs = [(conditional_word_00_given_word_10_symbol / conditional_word_01_given_word_10_symbol) * \
        (conditional_word_10_given_word_01_symbol / conditional_word_11_given_word_01_symbol)]

        lhs = lhs + [
            conditional_word_10_given_word_00_symbol,
            conditional_word_11_given_word_00_symbol,
            conditional_word_10_given_word_01_symbol,
            conditional_word_11_given_word_01_symbol,
            conditional_word_00_given_word_10_symbol,
            conditional_word_01_given_word_10_symbol,
            conditional_word_00_given_word_11_symbol,
            conditional_word_01_given_word_11_symbol
        ]

        rhs = rhs + [
            conditional_word_10_given_word_00,
            conditional_word_11_given_word_00,
            conditional_word_10_given_word_01,
            conditional_word_11_given_word_01,
            conditional_word_00_given_word_10,
            conditional_word_01_given_word_10,
            conditional_word_00_given_word_11,
            conditional_word_01_given_word_11
        ]
        analytics[url] = dict()

        solved = []
        try:
            for i in range(8):
                eqs = [l - r for l,r in zip(lhs[:i+1] + lhs[i+2:], rhs[:i+1] + rhs[i+2:])]
                solved_and_given = sym.nonlinsolve(eqs, [conditional_word_10_given_word_00_symbol,
                                                        conditional_word_11_given_word_00_symbol,
                                                        conditional_word_10_given_word_01_symbol,
                                                        conditional_word_11_given_word_01_symbol,
                                                        conditional_word_00_given_word_10_symbol,
                                                        conditional_word_01_given_word_10_symbol,
                                                        conditional_word_00_given_word_11_symbol,
                                                        conditional_word_01_given_word_11_symbol])
                solved.append(list(solved_and_given)[0][i])
        except:
            print('exception occured')
            print(url)
            analytics[url]['status'] = 'exception'
            continue



        conditional_word_10_given_word_00_solved = solved[0]
        conditional_word_11_given_word_00_solved = solved[1]
        conditional_word_10_given_word_01_solved = solved[2]
        conditional_word_11_given_word_01_solved = solved[3]
        conditional_word_00_given_word_10_solved = solved[4]
        conditional_word_01_given_word_10_solved = solved[5]
        conditional_word_00_given_word_11_solved = solved[6]
        conditional_word_01_given_word_11_solved = solved[7]
        
        
        # add the conditional probabilities (both generated and solved) to the dictionary
        analytics[url]['status'] = 'success'
        analytics[url]['conditional_word_00_given_word_10_solved'] = conditional_word_00_given_word_10_solved
        analytics[url]['conditional_word_01_given_word_10_solved'] = conditional_word_01_given_word_10_solved
        analytics[url]['conditional_word_00_given_word_11_solved'] = conditional_word_00_given_word_11_solved
        analytics[url]['conditional_word_01_given_word_11_solved'] = conditional_word_01_given_word_11_solved
        analytics[url]['conditional_word_10_given_word_00_solved'] = conditional_word_10_given_word_00_solved
        analytics[url]['conditional_word_11_given_word_00_solved'] = conditional_word_11_given_word_00_solved
        analytics[url]['conditional_word_10_given_word_01_solved'] = conditional_word_10_given_word_01_solved
        analytics[url]['conditional_word_11_given_word_01_solved'] = conditional_word_11_given_word_01_solved

        analytics[url]['conditional_word_00_given_word_10'] = conditional_word_00_given_word_10
        analytics[url]['conditional_word_01_given_word_10'] = conditional_word_01_given_word_10
        analytics[url]['conditional_word_00_given_word_11'] = conditional_word_00_given_word_11
        analytics[url]['conditional_word_01_given_word_11'] = conditional_word_01_given_word_11
        analytics[url]['conditional_word_10_given_word_00'] = conditional_word_10_given_word_00
        analytics[url]['conditional_word_11_given_word_00'] = conditional_word_11_given_word_00
        analytics[url]['conditional_word_10_given_word_01'] = conditional_word_10_given_word_01
        analytics[url]['conditional_word_11_given_word_01'] = conditional_word_11_given_word_01

        analytics[url]['inconsistency_word_00_given_word_10'] = abs(math.log(conditional_word_00_given_word_10_solved) - math.log(conditional_word_00_given_word_10))
        analytics[url]['inconsistency_word_01_given_word_10'] = abs(math.log(conditional_word_01_given_word_10_solved) - math.log(conditional_word_01_given_word_10))
        analytics[url]['inconsistency_word_00_given_word_11'] = abs(math.log(conditional_word_00_given_word_11_solved) - math.log(conditional_word_00_given_word_11))
        analytics[url]['inconsistency_word_01_given_word_11'] = abs(math.log(conditional_word_01_given_word_11_solved) - math.log(conditional_word_01_given_word_11))
        analytics[url]['inconsistency_word_10_given_word_00'] = abs(math.log(conditional_word_10_given_word_00_solved) - math.log(conditional_word_10_given_word_00))
        analytics[url]['inconsistency_word_11_given_word_00'] = abs(math.log(conditional_word_11_given_word_00_solved) - math.log(conditional_word_11_given_word_00))
        analytics[url]['inconsistency_word_10_given_word_01'] = abs(math.log(conditional_word_10_given_word_01_solved) - math.log(conditional_word_10_given_word_01))
        analytics[url]['inconsistency_word_11_given_word_01'] = abs(math.log(conditional_word_11_given_word_01_solved) - math.log(conditional_word_11_given_word_01))

        analytics[url]['avg_inconsistency'] = (analytics[url]['inconsistency_word_00_given_word_10'] + \
                                                analytics[url]['inconsistency_word_01_given_word_10'] + \
                                                analytics[url]['inconsistency_word_00_given_word_11'] + \
                                                analytics[url]['inconsistency_word_01_given_word_11'] + \
                                                analytics[url]['inconsistency_word_10_given_word_00'] + \
                                                analytics[url]['inconsistency_word_11_given_word_00'] + \
                                                analytics[url]['inconsistency_word_10_given_word_01'] + \
                                                analytics[url]['inconsistency_word_11_given_word_01']) / 8

    # dump analytics to a pickle file
    with open(args.analytics_filename, 'wb') as handle:
        pickle.dump(analytics, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    run()