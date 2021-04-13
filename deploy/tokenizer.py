#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: rossliang
# create time: 2021/4/13 5:19 下午


from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, help='input file')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-w', type=int, default=-1, help='worker size')
    args = parser.parse_args()
    return args


def main(args):
    pass


if __name__ == "__main__":
    pass
