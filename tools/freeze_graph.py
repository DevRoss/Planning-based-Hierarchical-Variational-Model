#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_graph', type=str, help='input graph, e.g. xxx.meta, xxx.pb, xxx.pbtxt')
    parser.add_argument('--input_checkpoint', type=str, help='the checkpoint path ')
    parser.add_argument('--output_node_names', type=str, default=-1, help='the output node names for graph, seperated by ,')
    parser.add_argument('--output_path', type=str, default=-1, help='the freeze graph output path')
    args = parser.parse_args()
    return args


def parse_node_names(str_node_names):
    return str_node_names.split(',')


def freeze(graph_path, ckpt_path, output_node_names, output_path):
    # meta_path = '../result/checkpoint/PHVM/best/best_9-8980.meta' # Your .meta file
    # output_node_names = ['sentence_level/infer/output_stop', 'sentence_level/infer/output_translate']    # Output nodes

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(graph_path, clear_devices=True)

        # Load weights
        saver.restore(sess, ckpt_path)

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open(output_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


def main(args):
    output_node_names = parse_node_names(args.output_node_names)
    freeze(args.input_graph, args.input_checkpoint, output_node_names, args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
