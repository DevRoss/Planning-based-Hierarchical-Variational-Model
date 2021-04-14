#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: rossliang
# create time: 2021/4/9 8:24 下午

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from argparse import ArgumentParser
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pb_path', type=str, help='input file')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-w', type=int, default=-1, help='worker size')
    args = parser.parse_args()
    return args


class PHVMOnline:
    def __init__(self, config):
        self.config = config
        self.val_input = None
        self.input_lens = None
        self.text = None
        self.slens = None
        self.category = None
        self.train_flag = None
        self.keep_prob = None
        self.start_token = None
        self.end_token = None
        self.output_stop = None
        self.output_translate = None
        self.load_model()

    def apply(self):
        pass

    def load_model(self):
        with tf.gfile.GFile(self.config['pb_path'], "rb") as pb:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(pb.read())
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(
                graph_def,
                name="",  # name可以自定义，修改name之后记得在下面的代码中也要改过来
            )
            self.val_input = self.graph.get_tensor_by_name('val_input:0')
            self.input_lens = self.graph.get_tensor_by_name('input_lens:0')
            self.text = self.graph.get_tensor_by_name('text:0')
            self.slens = self.graph.get_tensor_by_name('slens:0')
            self.category = self.graph.get_tensor_by_name('category:0')
            self.keep_prob = self.graph.get_tensor_by_name('Placeholder:0')
            self.train_flag = self.graph.get_tensor_by_name('Placeholder_1:0')
            self.output_stop = self.graph.get_tensor_by_name('sentence_level/infer/output_stop:0')
            self.output_translate = self.graph.get_tensor_by_name('sentence_level/infer/output_translate:0')

        # for op in self.graph.get_operations():
        #     print(op.name, op.values())  # 打印网络结构

    def infer(self, input_data):
        with tf.Session(graph=self.graph) as sess:
            feed_dict = {self.val_input: np.random.random_integers(0, 10, size=(3, 4)),
                         self.input_lens: np.array([4, 4, 4]),
                         self.slens: np.array([1] * 50),
                         self.category: np.random.random_integers(0, 2, size=(3,)),
                         self.train_flag: False,
                         self.keep_prob: 1
                         }
            stop, translations = sess.run((self.output_stop, self.output_translate), feed_dict=feed_dict)
            return self._agg_group(stop, translations)

    def _agg_group(self, stop, text):
        translation = []
        for gcnt, sent in zip(stop, text):
            sent = sent[:gcnt, :]
            desc = []
            for segId, seg in enumerate(sent):
                for wid in seg:
                    if wid == self.end_token:
                        break
                    elif wid == self.start_token:
                        continue
                    else:
                        desc.append(wid)
            translation.append(desc)
        max_len = 0
        for sent in translation:
            max_len = max(max_len, len(sent))
        for i, sent in enumerate(translation):
            translation[i] = [sent + [self.end_token] * (max_len - len(sent))]
        return translation


def main(args):
    xx = PHVMOnline(vars(args))
    import time
    start = time.time()
    xx.infer(None)
    print('cost {}ms'.format((time.time() - start)* 1000))


if __name__ == "__main__":
    args = parse_args()
    main(args)
