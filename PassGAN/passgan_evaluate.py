#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore all warning messages
import time
import pickle
import argparse
from pathlib import Path

import sys
sys.path.append(os.getcwd())  # Add current working directory to Python module search path

# Import necessary packages
import utils
import models
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Set log level to only show errors
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

import tensorflow.compat.v1 as tf  # Use TensorFlow 1.x in compatibility mode
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior

import tflib.plot  # Used for plotting training graphs

######### Password Strength Evaluation #########

def evaluate_run(args):
    # Ensure input_dir is a Path object
    input_dir = Path(args.input_dir)

    # Load character mapping files
    with open(input_dir / 'charmap.pickle', 'rb') as f:
        charmap = pickle.load(f)

    with open(input_dir / 'inv_charmap.pickle', 'rb') as f:
        inv_charmap = pickle.load(f)

    # Placeholder for real password inputs
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

    # Build the discriminator model
    disc_real_logits = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))

    # Apply sigmoid activation to limit the scores between 0 and 1
    disc_real = tf.nn.sigmoid(disc_real_logits)

    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.Saver()  # Restore the trained model
        saver.restore(session, args.checkpoint)

        # Load passwords from the input file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            passwords = f.read().splitlines()

        # Preprocess passwords
        passwords = [list(p)[:args.seq_length] for p in passwords]
        passwords = [[charmap.get(c, charmap['`']) for c in p] for p in passwords]  # Use '`' for unknown chars

        # Pad passwords to seq_length
        passwords_padded = []
        for p in passwords:
            if len(p) < args.seq_length:
                p += [charmap['`']] * (args.seq_length - len(p))
            passwords_padded.append(p)

        # Evaluate in batches
        scores = []
        for i in range(0, len(passwords_padded), args.batch_size):
            batch = passwords_padded[i:i+args.batch_size]
            batch_size_actual = len(batch)
            if len(batch) < args.batch_size:
                # Pad batch to batch_size
                batch += [[charmap['`']] * args.seq_length] * (args.batch_size - len(batch))
            feed_dict = {real_inputs_discrete: batch}
            disc_scores = session.run(disc_real, feed_dict=feed_dict)
            scores.extend(disc_scores[:batch_size_actual])

        # Save evaluation results
        with open(args.output, 'w', encoding='utf-8') as f:
            for pwd, score in zip(passwords, scores):
                pwd_str = ''.join([inv_charmap[idx] for idx in pwd if idx != charmap['`']])
                f.write(f"{pwd_str}\t{score}\n")

        print(f"Evaluation completed. Results saved to {args.output}")

###################################

# Display help information
def help():
    print("A Deep Learning Approach for Password Guessing.\n")
    print("List of arguments:\n")
    print("-h, --help              show this help message and exit")
    print("sample                  use the pretrained model to generate passwords")
    print("train                   train a model on a large dataset")
    print("evaluate                evaluate password strength using the discriminator")
    print("")
    print("Usage Examples:")
    print("passgan.py sample --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000")
    print("passgan.py train --output-dir pretrained --training-data data/train.txt")
    print("passgan.py evaluate --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --input-file passwords.txt --output scores.txt --batch-size 64")

def main(args=None):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action='store_true', help="show this help message and exit")
    subparsers = parser.add_subparsers(dest="cmd")

    # Evaluation command arguments
    subp_evaluate = subparsers.add_parser("evaluate", help='evaluate password strength using the discriminator', add_help=False)

    # Evaluation command arguments
    subp_evaluate.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    subp_evaluate.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for evaluation. Expects a .ckpt file.')

    subp_evaluate.add_argument('--input-file', '-f',
                        required=True,
                        dest='input_file',
                        help='File containing passwords to evaluate (one per line).')

    subp_evaluate.add_argument('--output', '-o',
                        default='scores.txt',
                        help='File path to save evaluation results (default: scores.txt)')

    subp_evaluate.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')

    subp_evaluate.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')

    subp_evaluate.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the discriminator. Use the same value that you did for training (default: 128)')

    parsed_args = parser.parse_args(args)

    if parsed_args.cmd == "evaluate":
        if not Path(parsed_args.input_dir).is_dir():
            parser.error(f'"{parsed_args.input_dir}" folder doesn\'t exist')

        if not Path(f'{parsed_args.checkpoint}.meta').is_file():
            parser.error(f'"{parsed_args.checkpoint}.meta" file doesn\'t exist')

        input_dir = Path(parsed_args.input_dir)

        if not (input_dir / 'charmap.pickle').is_file():
            parser.error(f'charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        if not (input_dir / 'inv_charmap.pickle').is_file():
            parser.error(f'inv_charmap.pickle doesn\'t exist in {parsed_args.input_dir}, are you sure that directory is a trained model directory?')

        evaluate_run(parsed_args)
    elif parsed_args.help or (not(parsed_args.cmd)):
        help()
        exit()

if __name__ == "__main__":
    main()
