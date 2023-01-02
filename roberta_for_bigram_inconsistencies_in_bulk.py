from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import copy
import sympy as sym
import numpy as np
import pickle
from argparse import ArgumentParser

from scipy.special import rel_entr
from tqdm import tqdm


def run():
    parser = ArgumentParser()

    parser.add_argument("--no_samples", type=int, default=10000)
    parser.add_argument("--model_name", type=str, default='roberta-base')
    parser.add_argument("--analytics_filename", type=str, default='roberta-base-on-bigrams.pkl')
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(args.model_name)
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
        conditional_word_10_given_word_00_normalized = conditional_word_10_given_word_00 / (conditional_word_10_given_word_00 + conditional_word_11_given_word_00)
        conditional_word_11_given_word_00_normalized = 1 - conditional_word_10_given_word_00_normalized
        conditional_word_10_given_word_01_normalized = conditional_word_10_given_word_01 / (conditional_word_10_given_word_01 + conditional_word_11_given_word_01)
        conditional_word_11_given_word_01_normalized = 1 - conditional_word_10_given_word_01_normalized
        conditional_word_00_given_word_10_normalized = conditional_word_00_given_word_10 / (conditional_word_00_given_word_10 + conditional_word_01_given_word_10)
        conditional_word_01_given_word_10_normalized = 1 - conditional_word_00_given_word_10_normalized
        conditional_word_00_given_word_11_normalized = conditional_word_00_given_word_11 / (conditional_word_00_given_word_11 + conditional_word_01_given_word_11)
        conditional_word_01_given_word_11_normalized = 1 - conditional_word_00_given_word_11_normalized


        joint_00 = sym.symbols('joint_00') # the joint probability of x_0 = word_00 and x_1 = word_10
        joint_01 = sym.symbols('joint_01') # the joint probability of x_0 = word_00 and x_1 = word_11
        joint_10 = sym.symbols('joint_10') # the joint probability of x_0 = word_01 and x_1 = word_10
        joint_11 = sym.symbols('joint_11') # the joint probability of x_0 = word_01 and x_1 = word_11


        conditional_word_00_given_word_10_normalized_symbol = sym.symbols('conditional_word_00_given_word_10_normalized')
        conditional_word_00_given_word_11_normalized_symbol = sym.symbols('conditional_word_00_given_word_11_normalized')
        conditional_word_10_given_word_00_normalized_symbol = sym.symbols('conditional_word_10_given_word_00_normalized')
        conditional_word_10_given_word_01_normalized_symbol = sym.symbols('conditional_word_10_given_word_01_normalized')

        lhs_template = [
            joint_00 + joint_01 + joint_10 + joint_11,
            joint_00 / (joint_00 + joint_10),
            joint_01 / (joint_01 + joint_11),
            joint_00 / (joint_00 + joint_01),
            joint_10 / (joint_10 + joint_11)]

        rhs_template = [1, \
            conditional_word_00_given_word_10_normalized_symbol, \
            conditional_word_00_given_word_11_normalized_symbol, \
            conditional_word_10_given_word_00_normalized_symbol, \
            conditional_word_10_given_word_01_normalized_symbol]

        lhs_template.append(conditional_word_00_given_word_10_normalized_symbol)
        rhs_template.append(conditional_word_00_given_word_10_normalized)
        lhs_template.append(conditional_word_00_given_word_11_normalized_symbol)
        rhs_template.append(conditional_word_00_given_word_11_normalized)
        lhs_template.append(conditional_word_10_given_word_00_normalized_symbol)
        rhs_template.append(conditional_word_10_given_word_00_normalized)
        lhs_template.append(conditional_word_10_given_word_01_normalized_symbol)
        rhs_template.append(conditional_word_10_given_word_01_normalized)

        # calculate conditional_word_00_given_word_10_normalized given the other conditionals
        # remove the symbol from the lhs and rhs
        lhs = copy.deepcopy(lhs_template)
        rhs = copy.deepcopy(rhs_template)
        for i in range(len(lhs)):
            if lhs[i] == conditional_word_00_given_word_10_normalized_symbol:
                lhs.pop(i)
                rhs.pop(i)
                break
        eqs = [l - r for l,r in zip(lhs, rhs)]

        solved_values = sym.solve(eqs, [conditional_word_00_given_word_10_normalized_symbol, \
            conditional_word_00_given_word_11_normalized_symbol, \
            conditional_word_10_given_word_00_normalized_symbol, \
            conditional_word_10_given_word_01_normalized_symbol, \
            joint_00, joint_01, joint_10, joint_11])

        conditional_word_00_given_word_10_normalized_solved = float(solved_values[0][0])
        conditional_word_01_given_word_10_normalized_solved = 1 - conditional_word_00_given_word_10_normalized_solved

        # calculate conditional_word_00_given_word_11_normalized given the other conditionals
        # remove the symbol from the lhs and rhs
        lhs = copy.deepcopy(lhs_template)
        rhs = copy.deepcopy(rhs_template)
        for i in range(len(lhs)):
            if lhs[i] == conditional_word_00_given_word_11_normalized_symbol:
                lhs.pop(i)
                rhs.pop(i)
                break
        eqs = [l - r for l,r in zip(lhs, rhs)]
        solved_values = sym.solve(eqs, [conditional_word_00_given_word_10_normalized_symbol, \
            conditional_word_00_given_word_11_normalized_symbol, \
            conditional_word_10_given_word_00_normalized_symbol, \
            conditional_word_10_given_word_01_normalized_symbol, \
            joint_00, joint_01, joint_10, joint_11])
        
        conditional_word_00_given_word_11_normalized_solved = float(solved_values[0][1])
        conditional_word_01_given_word_11_normalized_solved = 1 - conditional_word_00_given_word_11_normalized_solved

        # calculate conditional_word_10_given_word_00_normalized given the other conditionals
        # remove the symbol from the lhs and rhs
        lhs = copy.deepcopy(lhs_template)
        rhs = copy.deepcopy(rhs_template)
        for i in range(len(lhs)):
            if lhs[i] == conditional_word_10_given_word_00_normalized_symbol:
                lhs.pop(i)
                rhs.pop(i)
                break
        eqs = [l - r for l,r in zip(lhs, rhs)]
        solved_values = sym.solve(eqs, [conditional_word_00_given_word_10_normalized_symbol, \
            conditional_word_00_given_word_11_normalized_symbol, \
            conditional_word_10_given_word_00_normalized_symbol, \
            conditional_word_10_given_word_01_normalized_symbol, \
            joint_00, joint_01, joint_10, joint_11])

        conditional_word_10_given_word_00_normalized_solved = float(solved_values[0][2])
        conditional_word_11_given_word_00_normalized_solved = 1 - conditional_word_10_given_word_00_normalized_solved

        # calculate conditional_word_10_given_word_01_normalized given the other conditionals
        # remove the symbol from the lhs and rhs
        lhs = copy.deepcopy(lhs_template)
        rhs = copy.deepcopy(rhs_template)
        for i in range(len(lhs)):
            if lhs[i] == conditional_word_10_given_word_01_normalized_symbol:
                lhs.pop(i)
                rhs.pop(i)
                break
        eqs = [l - r for l,r in zip(lhs, rhs)]
        solved_values = sym.solve(eqs, [conditional_word_00_given_word_10_normalized_symbol, \
            conditional_word_00_given_word_11_normalized_symbol, \
            conditional_word_10_given_word_00_normalized_symbol, \
            conditional_word_10_given_word_01_normalized_symbol, \
            joint_00, joint_01, joint_10, joint_11])
        
        conditional_word_10_given_word_01_normalized_solved = float(solved_values[0][3])
        conditional_word_11_given_word_01_normalized_solved = 1 - conditional_word_10_given_word_01_normalized_solved


        conditionals_given_word_10 = [conditional_word_00_given_word_10_normalized, conditional_word_01_given_word_10_normalized]
        conditionals_given_word_10_solved = [conditional_word_00_given_word_10_normalized_solved, conditional_word_01_given_word_10_normalized_solved] 
        conditionals_given_word_11 = [conditional_word_00_given_word_11_normalized, conditional_word_01_given_word_11_normalized]
        conditionals_given_word_11_solved = [conditional_word_00_given_word_11_normalized_solved, conditional_word_01_given_word_11_normalized_solved]
        conditionals_given_word_00 = [conditional_word_10_given_word_00_normalized, conditional_word_11_given_word_00_normalized]
        conditionals_given_word_00_solved = [conditional_word_10_given_word_00_normalized_solved, conditional_word_11_given_word_00_normalized_solved]
        conditionals_given_word_01 = [conditional_word_10_given_word_01_normalized, conditional_word_11_given_word_01_normalized]
        conditionals_given_word_01_solved = [conditional_word_10_given_word_01_normalized_solved, conditional_word_11_given_word_01_normalized_solved]

        rel_entr_given_word_10 = rel_entr(conditionals_given_word_10, conditionals_given_word_10_solved)
        kl_div_given_word_10 = np.sum(rel_entr_given_word_10)

        rel_entr_given_word_11 = rel_entr(conditionals_given_word_11, conditionals_given_word_11_solved)
        kl_div_given_word_11 = np.sum(rel_entr_given_word_11)

        rel_entr_given_word_00 = rel_entr(conditionals_given_word_00, conditionals_given_word_00_solved)
        kl_div_given_word_00 = np.sum(rel_entr_given_word_00)

        rel_entr_given_word_01 = rel_entr(conditionals_given_word_01, conditionals_given_word_01_solved)
        kl_div_given_word_01 = np.sum(rel_entr_given_word_01)

        kl_div_avg = (kl_div_given_word_10 + kl_div_given_word_11 + kl_div_given_word_00 + kl_div_given_word_01) / 4

        # add the conditional probabilities (both generated and solved) to the dictionary
        analytics[url] = dict()
        analytics[url]['kl_div_avg'] = kl_div_avg
        analytics[url]['kl_div_given_word_10'] = kl_div_given_word_10
        analytics[url]['kl_div_given_word_11'] = kl_div_given_word_11
        analytics[url]['kl_div_given_word_00'] = kl_div_given_word_00
        analytics[url]['kl_div_given_word_01'] = kl_div_given_word_01

        analytics[url]['conditional_word_00_given_word_10_normalized'] = conditional_word_00_given_word_10_normalized
        analytics[url]['conditional_word_01_given_word_10_normalized'] = conditional_word_01_given_word_10_normalized
        analytics[url]['conditional_word_00_given_word_11_normalized'] = conditional_word_00_given_word_11_normalized
        analytics[url]['conditional_word_01_given_word_11_normalized'] = conditional_word_01_given_word_11_normalized
        analytics[url]['conditional_word_10_given_word_00_normalized'] = conditional_word_10_given_word_00_normalized
        analytics[url]['conditional_word_11_given_word_00_normalized'] = conditional_word_11_given_word_00_normalized
        analytics[url]['conditional_word_10_given_word_01_normalized'] = conditional_word_10_given_word_01_normalized
        analytics[url]['conditional_word_11_given_word_01_normalized'] = conditional_word_11_given_word_01_normalized

        analytics[url]['conditional_word_00_given_word_10_normalized_solved'] = conditional_word_00_given_word_10_normalized_solved
        analytics[url]['conditional_word_01_given_word_10_normalized_solved'] = conditional_word_01_given_word_10_normalized_solved
        analytics[url]['conditional_word_00_given_word_11_normalized_solved'] = conditional_word_00_given_word_11_normalized_solved
        analytics[url]['conditional_word_01_given_word_11_normalized_solved'] = conditional_word_01_given_word_11_normalized_solved
        analytics[url]['conditional_word_10_given_word_00_normalized_solved'] = conditional_word_10_given_word_00_normalized_solved
        analytics[url]['conditional_word_11_given_word_00_normalized_solved'] = conditional_word_11_given_word_00_normalized_solved
        analytics[url]['conditional_word_10_given_word_01_normalized_solved'] = conditional_word_10_given_word_01_normalized_solved
        analytics[url]['conditional_word_11_given_word_01_normalized_solved'] = conditional_word_11_given_word_01_normalized_solved

        analytics[url]['conditional_word_00_given_word_10'] = conditional_word_00_given_word_10
        analytics[url]['conditional_word_01_given_word_10'] = conditional_word_01_given_word_10
        analytics[url]['conditional_word_00_given_word_11'] = conditional_word_00_given_word_11
        analytics[url]['conditional_word_01_given_word_11'] = conditional_word_01_given_word_11
        analytics[url]['conditional_word_10_given_word_00'] = conditional_word_10_given_word_00
        analytics[url]['conditional_word_11_given_word_00'] = conditional_word_11_given_word_00
        analytics[url]['conditional_word_10_given_word_01'] = conditional_word_10_given_word_01
        analytics[url]['conditional_word_11_given_word_01'] = conditional_word_11_given_word_01

    # dump analytics to a pickle file
    with open(args.analytics_filename, 'wb') as handle:
        pickle.dump(analytics, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    run()