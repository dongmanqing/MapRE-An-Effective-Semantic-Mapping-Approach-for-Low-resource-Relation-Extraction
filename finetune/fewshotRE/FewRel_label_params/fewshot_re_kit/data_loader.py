import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import functools

from transformers import AutoTokenizer


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        # for j in range(Q_na):
        #     cur_class = np.random.choice(na_classes, 1, False)[0]
        #     index = np.random.choice(
        #             list(range(len(self.json_data[cur_class]))),
        #             1, False)[0]
        #     word, pos1, pos2, mask = self.__getraw__(
        #             self.json_data[cur_class][index])
        #     word = torch.tensor(word).long()
        #     pos1 = torch.tensor(pos1).long()
        #     pos2 = torch.tensor(pos2).long()
        #     mask = torch.tensor(mask).long()
        #     self.__additem__(query_set, word, pos1, pos2, mask)
        # query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000


class FewRelDatasetLabelEmbedding(data.Dataset):
    def __init__(self, name, pid2name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name, 'r'))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        descs = []
        for i, class_name in enumerate(target_classes):
            if class_name in self.pid2name:
                desc = self.pid2name[class_name][0]
            else:
                desc = class_name.replace('_', ' ')

            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            descs.append(desc)
            query_label += [i] * self.Q

        # NA
        # for j in range(Q_na):
        #     cur_class = np.random.choice(na_classes, 1, False)[0]
        #     index = np.random.choice(
        #             list(range(len(self.json_data[cur_class]))),
        #             1, False)[0]
        #     word, pos1, pos2, mask = self.__getraw__(
        #             self.json_data[cur_class][index])
        #     word = torch.tensor(word).long()
        #     pos1 = torch.tensor(pos1).long()
        #     pos2 = torch.tensor(pos2).long()
        #     mask = torch.tensor(mask).long()
        #     self.__additem__(query_set, word, pos1, pos2, mask)
        # query_label += [self.N] * Q_na

        return support_set, query_set, query_label, descs

    def __len__(self):
        return 1000000000


class TestSet(data.Dataset):
    def __init__(self, dat_path, sen_enc, n, k):
        self.data = json.load(open(dat_path))
        self.sentence_encoder = sen_enc
        self.n = n
        self.k = k

    def _load_example(self, example):
        tokens, pos1, pos2, mask = self.sentence_encoder.tokenize(example['tokens'], example['h'][2][0],
                                                                  example['t'][2][0])
        return tokens, pos1, pos2, mask

    @staticmethod
    def _add_example(d, tokens, pos1, pos2, mask):
        tokens = torch.tensor(tokens).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(tokens)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, idx):
        task = self.data[idx]

        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        support_label = []
        query_label = [0]

        meta_train = task['meta_train']
        meta_test = task['meta_test']
        # relation_set = task['relation']

        tokens, pos1, pos2, mask = self._load_example(meta_test)
        self._add_example(query_set, tokens, pos1, pos2, mask)

        for current_n in range(self.n):
            for current_k in range(self.k):
                tokens, pos1, pos2, mask = self._load_example(meta_train[current_n][current_k])
                self._add_example(support_set, tokens, pos1, pos2, mask)
            support_label += [current_n] * self.k

        return support_set, query_set, query_label

    def __len__(self):
        return len(self.data)


class TestSetWithLabel(data.Dataset):
    def __init__(self, pid2name, dat_path, sen_enc, n, k):
        self.data = json.load(open(dat_path))
        self.pid2name = json.load(open(pid2name, 'r'))
        self.sentence_encoder = sen_enc
        self.n = n
        self.k = k

    def _load_example(self, example):
        tokens, pos1, pos2, mask = self.sentence_encoder.tokenize(example['tokens'], example['h'][2][0],
                                                                  example['t'][2][0])
        return tokens, pos1, pos2, mask

    @staticmethod
    def _add_example(d, tokens, pos1, pos2, mask):
        tokens = torch.tensor(tokens).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(tokens)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, idx):
        task = self.data[idx]

        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        support_label = []
        query_label = [0]

        meta_train = task['meta_train']
        meta_test = task['meta_test']
        relation_set = task['relation']

        tokens, pos1, pos2, mask = self._load_example(meta_test)
        self._add_example(query_set, tokens, pos1, pos2, mask)

        descs = []

        for current_n in range(self.n):
            r = relation_set[current_n]
            if r in self.pid2name:
                desc = self.pid2name[r][0]
            else:
                desc = r.replace('_', ' ')
            descs.append(desc)
            for current_k in range(self.k):
                tokens, pos1, pos2, mask = self._load_example(meta_train[current_n][current_k])
                self._add_example(support_set, tokens, pos1, pos2, mask)
            support_label += [current_n] * self.k

        return support_set, query_set, query_label, descs

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, descs = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def collate_fn_label(data, tokenizer):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, descs = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    tmp = []
    for d in descs:
        tmp += d

    batch_desc = tokenizer(tmp, max_length=500, truncation=True, padding=True, return_tensors='pt')
    return batch_support, batch_query, batch_label, batch_desc


def get_loader(name, encoder, N, K, Q, batch_size,
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


def get_loader_label(name, pid2name, encoder, N, K, Q, batch_size, tokenizer_name, num_workers=8, na_rate=0, root='./data'):
    dataset = FewRelDatasetLabelEmbedding(name, pid2name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=functools.partial(collate_fn_label, tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)))
    return iter(data_loader)


def get_loader_test(name, encoder, N, K, Q, batch_size,
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = TestSet(os.path.join(root, '{}.json'.format(name)), encoder, N, K)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


def get_loader_test_label(name, pid2name, encoder, N, K, Q, batch_size, tokenizer_name,
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = TestSetWithLabel(pid2name, os.path.join(root, '{}.json'.format(name)), encoder, N, K)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=functools.partial(collate_fn_label, tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)))
    return iter(data_loader)


class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=0, collate_fn=collate_fn_pair, na_rate=0, root='data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


