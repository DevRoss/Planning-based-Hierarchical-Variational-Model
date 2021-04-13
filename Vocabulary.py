import ctypes
import json
import multiprocessing.sharedctypes as mpsc
import pickle

import numpy as np

import Config


class Vocabulary:
    # def __init__(self):
    #     self.config = Config.config
    #
    #     category = list(pickle.load(open(self.config.category_file, "rb")))
    #     featCate = list(pickle.load(open(self.config.feat_key_file, "rb")))
    #     featVal = list(pickle.load(open(self.config.feat_val_file, "rb")))
    #     cateFK2val = pickle.load(open(self.config.cateFK2val_file, "rb"))
    #     self.cateFK2val = cateFK2val
    #
    #     self.id2category = category
    #     self.category2id = dict(zip(self.id2category, range(len(self.id2category))))
    #
    #     self.id2featCate = ["<MARKER>", "<SENT>"] + featCate
    #     self.featCate2id = dict(zip(self.id2featCate, range(len(self.id2featCate))))
    #
    #     self.id2type = ["<GENERAL>"] + featCate
    #     self.type2id = dict(zip(self.id2type, range(len(self.id2type))))
    #
    #     self.id2featVal = ["<S>", "<ADJ>"] + featVal
    #     self.featVal2id = dict(zip(self.id2featVal, range(len(self.id2featVal))))
    #
    #     self.id2word = ["<S>", "</S>", 0] + [0] * len(featVal)
    #     self.id2vec = [0] * (3 + len(featVal))
    #     nxt = 3
    #     with open(self.config.wordvec_file, "r") as file:
    #         for _ in range(self.config.skip_cnt):
    #             file.readline()
    #         for line in file:
    #             line = line.split(" ")
    #             word = line[0]
    #             vec = [eval(i) for i in line[1:]]
    #             if word in featVal:
    #                 self.id2word[nxt] = word
    #                 self.id2vec[nxt] = vec
    #                 nxt += 1
    #             elif word == "<UNK>":
    #                 self.id2word[2] = "<UNK>"
    #                 self.id2vec[2] = vec
    #             else:
    #                 self.id2word.append(word)
    #                 self.id2vec.append(vec)
    #     for val in featVal:
    #         if val not in self.id2word:
    #             self.id2word.append(val)
    #             self.id2vec[nxt] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim,)))
    #             nxt += 1
    #     assert nxt == len(featVal) + 3
    #     self.keywords_cnt = nxt
    #     fcnt = 2
    #     if "<UNK>" not in self.id2word:
    #         self.id2word[2] = "<UNK>"
    #         fcnt += 1
    #     for i in range(fcnt):
    #         self.id2vec[i] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim,)))
    #     self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
    #
    #     self.table = [self.featCate2id, self.featVal2id, self.word2id, self.type2id]
    #
    #     self.start_token = 0
    #     self.end_token = 1

    def __init__(self):
        self.config = Config.config

        category = list(pickle.load(open(self.config.category_file, "rb")))
        featKey = list(pickle.load(open(self.config.feat_key_file, "rb")))
        featVal = list(pickle.load(open(self.config.feat_val_file, "rb")))
        cateFK2val = pickle.load(open(self.config.cateFK2val_file, "rb"))
        with open(self.config.category_file.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
            json.dump(category, f, ensure_ascii=False)
        with open(self.config.feat_key_file.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
            json.dump(featKey, f, ensure_ascii=False)
        with open(self.config.feat_val_file.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
            json.dump(featVal, f, ensure_ascii=False)
        with open(self.config.cateFK2val_file.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
            json.dump(cateFK2val, f, ensure_ascii=False)

        # feat_np, keys, goods = self.load_word_vector(self.config.wordvec_file)
        feat_np, keys, goods = self.load_word_vector(self.config.ailab_word2vec_file)

        # 构造词向量字典
        spec_tokens = ["<S>", "</S>", '<UNK>', "<MARKER>", "<SENT>", "<GENERAL>", "<ADJ>"]

        self.id2type = ["<GENERAL>"] + featKey
        self.type2id = dict(zip(self.id2type, range(len(self.id2type))))

        unk_tokens = self._collect_unk_token(feat_np, cateFK2val, featKey, featVal, category)
        extra_tokens = spec_tokens + unk_tokens
        self.id2word = [None] * len(keys)
        for k, index in keys.items():
            self.id2word[index] = k
        self.id2word += extra_tokens
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.id2vec = feat_np
        self.id2vec = np.concatenate([self.id2vec, np.random.uniform(low=-0.1, high=0.1, size=(len(extra_tokens), self.config.word_dim))], axis=0)
        self.keywords_cnt = len(self.id2word)

        self.id2category = self.id2word
        self.category2id = self.word2id


        self.table = [self.word2id, self.word2id, self.word2id, self.id2type]

        self.start_token = self.id2word.index('<S>')
        self.end_token = self.id2word.index('</S>')

    def _collect_unk_token(self, w2v, cateFK2val, featCate, featVal, category):
        unk_set = set()
        for k, v in cateFK2val.items():
            if k not in w2v:
                unk_set.add(k)
            for sv in v:
                if sv not in w2v:
                    unk_set.add(sv)

        for cate in featCate:
            if cate not in w2v:
                unk_set.add(cate)

        for v in featVal:
            if v not in w2v:
                unk_set.add(v)

        for cate in category:
            if cate not in w2v:
                unk_set.add(cate)
        return sorted(unk_set)

    def load_word_vector(self, feat_path):
        with open(feat_path, "rb") as f:
            tmp = pickle.load(f)
            feat = tmp['feat']
            keys = tmp['keys']
            goods = tmp['goods']
            shm_arr = mpsc.Array(ctypes.c_double, feat.size, lock=False)
            feat_np = np.frombuffer(shm_arr, dtype=feat.dtype).reshape(feat.shape)
            np.copyto(feat_np, feat)
        return feat_np, keys, goods

    def lookup(self, word, tpe):
        """
        :param word:
        :param tpe: 0 for featKey
                    1 for featVal
                    2 for word
                    3 for type
        :return:
        """
        if tpe == 2:
            return self.table[tpe].get(word, self.table[tpe]["<UNK>"])
        else:
            return self.table[tpe][word]
