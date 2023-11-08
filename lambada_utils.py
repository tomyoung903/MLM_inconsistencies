
ENDING_PUNCTUATIONS = ',!.:;?'

'''Get the first word from each of the given options. Return the words.'''
def get_words_from_options(options):
    # if a punctuation can be found in the option, get the word before the punctuation
    words = []
    for option in options:
        # find the punctuation
        for i in range(len(option)):
            if option[i] in ENDING_PUNCTUATIONS:
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



def get_word_from_option(option):
    '''Get the first word from the given option. Return the word.'''
    found = False
    # if a punctuation can be found in the option, get the word before the punctuation
    for i in range(len(option)):
        if option[i] in ENDING_PUNCTUATIONS:
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



def get_word_punc_pairs(options):
    '''given a list of options (completions by the LLM), return a list of word-punc pairs'''
    # show(options)
    # if a punctuation can be found in the option, get the word before the punctuation
    words = []
    for option in options:
        # find the punctuation
        for i in range(len(option)):
            if option[i] in ENDING_PUNCTUATIONS:
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



def remove_pad(options):
    '''given a list of options (completions by the LLM), remove the <pad>'''
    # if the word starts with <pad>, remove the <pad>
    options = [option[5:] if option.startswith("<pad>") else option for option in options]
    return options



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



def before_first_punc(options):
    options_return = []
    for option in options:
        for i in range(len(option)):
            if option[i] in ENDING_PUNCTUATIONS_IDS_LIST:
                options_return.append(option[:i+1])
                break
    return options_return