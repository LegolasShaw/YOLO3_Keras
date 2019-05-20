# coding: utf-8
# @author: Shaw
# @datetime: 2019-05-16 16:10
# @Name: convert.py

import argparse
from collections import defaultdict
import io

parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')

parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')

parser.add_argument('-p', '--plot_model',
                    help='Plot generated Keras model and save as image.',
                    action='store_true')

parser.add_argument('-w', '--weights_only',
                    help='Save as Keras weights file instead of model file.',
                    action='store_true')


def unique_config_section(config_file):
    """
    :param config_file:
    :return:
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()

    with open(config_file) as file:
        pass
