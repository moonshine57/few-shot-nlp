import os
import itertools
import collections
import json
from collections import defaultdict

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from dataloader.utils import tprint
from transformers import  BertTokenizer

def load_dataset(dataset_name,dataset_path,args):
    if dataset_name == 'reuters':
        train_classes, val_classes, test_classes = _get_reuters_classes()
    elif dataset_name == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes()
    elif dataset_name == '20newsgroup':
        train_classes, val_classes, test_classes = _get_20newsgroup_classes()
    elif dataset_name == 'fewrel':
        train_classes, val_classes, test_classes = _get_fewrel_classes()
    elif dataset_name == 'amazon':
        train_classes, val_classes, test_classes = _get_amazon_classes()
    elif dataset_name == 'event':
        train_classes, val_classes, test_classes = _get_event_classes()
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[reuters,huffpost,20newsgroup,fewrel,amazon]')
        
    tprint('Loading data')
    all_data = _load_json(dataset_path,args)
    
    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes,args)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_bert_features(train_data,args)
    val_data = _data_to_bert_features(val_data,args)
    test_data = _data_to_bert_features(test_data,args)

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool
    
    return train_data, val_data, test_data
    
    
def _get_20newsgroup_classes():
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
            'talk.politics.mideast': 0,
            'sci.space': 1,
            'misc.forsale': 2,
            'talk.politics.misc': 3,
            'comp.graphics': 4,
            'sci.crypt': 5,
            'comp.windows.x': 6,
            'comp.os.ms-windows.misc': 7,
            'talk.politics.guns': 8,
            'talk.religion.misc': 9,
            'rec.autos': 10,
            'sci.med': 11,
            'comp.sys.mac.hardware': 12,
            'sci.electronics': 13,
            'rec.sport.hockey': 14,
            'alt.atheism': 15,
            'rec.motorcycles': 16,
            'comp.sys.ibm.pc.hardware': 17,
            'rec.sport.baseball': 18,
            'soc.religion.christian': 19,
        }

    train_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['sci', 'rec']:
            train_classes.append(label_dict[key])

    val_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['comp']:
            val_classes.append(label_dict[key])

    test_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
            test_classes.append(label_dict[key])

    return train_classes, val_classes, test_classes
    
    

def _get_reuters_classes():
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(15))
    val_classes = list(range(15,20))
    test_classes = list(range(20,31))

    return train_classes, val_classes, test_classes

def _get_event_classes():
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(20))
    val_classes = [20,21,22,23,24]
    test_classes = list(range(25,35))

    return train_classes, val_classes, test_classes

def _get_huffpost_classes():
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(20))
    val_classes = list(range(20,25))
    test_classes = list(range(25,41))

    return train_classes, val_classes, test_classes

def _get_fewrel_classes():
    '''
        @return list of classes associated with each split
    '''
    # head=WORK_OF_ART validation/test split
    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    return train_classes, val_classes, test_classes

def _get_amazon_classes():
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }

    train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
    val_classes = [1, 22, 23, 6, 9]
    test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

    return train_classes, val_classes, test_classes


def _load_json(path,args):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1
            
            if args.dataset == 'event':
                item = {
                    'label': int(row['label']),
                    'text': row['text'][:500],  # truncate the text to 500 tokens
                    'category':int(row['category'])
                }
            else:
                item = {
                    'label': int(row['label']),
                    'text': row['text'][:500] # truncate the text to 500 tokens
                }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        tprint(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data
    
def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words

def _meta_split(all_data, train_classes, val_classes, test_classes,args):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    if args.dataset == 'event':
        for example in all_data:
            if example['category'] in train_classes:
                train_data.append(example)
            if example['category'] in val_classes:
                val_data.append(example)
            if example['category'] in test_classes:
                test_data.append(example)
    else:
        for example in all_data:
            if example['label'] in train_classes:
                train_data.append(example)
            if example['label'] in val_classes:
                val_data.append(example)
            if example['label'] in test_classes:
                test_data.append(example)

    return train_data, val_data, test_data


def _data_to_bert_features(data,args,model_name_or_path = 'bert-base-uncased',do_lower_case = True):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    
    #if args.shot == 5:
        #max_bert_len = 126
    #if args.shot == 1:
    max_bert_len = 510
    # convert each token to its corresponding id
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
    cls_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
    input_ids = []
    sentences_lengths = []
    input_masks = []
    token_type_ids = []
    max_len = 0
    for i in range(len(data)):
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(data[i]['text'])))
        tokens = tokens[:max_bert_len]
        if len(tokens)>max_len:
            max_len = len(tokens)
        input_id = [cls_token_id] + tokens + [sep_token_id]
        sentences_lengths.append(len(input_id))
        input_ids.append(input_id)
    
    new_input_ids = []
    for j in input_ids:
        token_type_id = [0] * len(j)
        input_mask = [1] * len(j)

        # Zero-pad up to the sequence length. BERT: Pad to the right
        padding = [0] * (max_len + 2 - len(j))
        new_input_id = j + padding
        new_input_ids.append(new_input_id)
        token_type_id += padding
        token_type_ids.append(token_type_id)
        input_mask += padding
        input_masks.append(input_mask)

        
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)
    doc_category = np.array([x['category'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)


    features = {
        'input_ids': np.asarray(new_input_ids),
        'input_mask': np.asarray(input_masks),
        'token_type_ids':np.asarray(token_type_ids),
        'sentences_lengths':np.asarray(sentences_lengths),
        'label': doc_label,
        'raw': raw,
        'category':doc_category
    }

    return features

def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list