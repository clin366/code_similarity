import Levenshtein as L
import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
import re
import tensorflow_hub as hub

""" 
This project is to detect the similarity between two python codes,
using the Levenshtein ratio similarity and Universal Encoder similarity.
"""

def read_code(code_path):
    sentences = []
    with open(code_path, 'r') as fi:
        line = fi.readline()

        block_annotation = False

        while line:
            add_line = line.strip()

            if add_line:
                if block_annotation:
                    if add_line.endswith('"""'):
    #                     print("block_annotation: " + add_line)
                        block_annotation = False

                elif not block_annotation:
    #                 print("not block_annotation: " + add_line)
                    if not add_line.startswith('#'):
                        if add_line.startswith('"""'):
    #                         print("start_with:" + add_line)
                            block_annotation = True
                        else:
                            sentences.append(line.strip())
            line = fi.readline()
    
    return sentences

def cal_similarity(sentences, pair_sent):
    ratio_list = []
    for line in sentences:
        max_ratio = None
        for pair_line in pair_sent:
            L_ratio = L.ratio(line, pair_line)

            if (max_ratio is None) or (L_ratio > max_ratio):
                max_ratio = L_ratio
        ratio_list.append(max_ratio)
    return ratio_list


## functions to calculate similarity based on universal encoder
def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
    message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages_})
    return message_embeddings_

def get_embedding(sentences, embed):
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        sentences_embeddings = run_and_plot(session, similarity_input_placeholder, sentences,
                   similarity_message_encodings)
    return sentences_embeddings

def cal_unicode_similarity(sentences, pair_sent):
    ratio_list = []
    for line in sentences:
        max_ratio = None
        for pair_line in pair_sent:
            corr = np.inner(line, pair_line)

            if (max_ratio is None) or (corr > max_ratio):
                max_ratio = corr
        ratio_list.append(max_ratio)
    return ratio_list




def get_similarity(first_file, second_file):
    sentences = read_code(first_file)
    pair_sent = read_code(second_file)
    sim_value = cal_similarity(pair_sent, sentences)
    sim_ratio = sum(sim_value)/len(sim_value)
    print("sim_ratio: " + str(sim_ratio))
    rev_sim_value = cal_similarity(sentences, pair_sent)
    rev_sim_ratio = sum(rev_sim_value)/len(rev_sim_value)
    print("rev_sim_ratio: " + str(rev_sim_ratio))
    print("avg_sim_ratio:" + str((sim_ratio + rev_sim_ratio) / 2 ))

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    sentences_embeddings = get_embedding(sentences, embed)
    pair_sent_embeddings = get_embedding(pair_sent, embed)

    uni_value = cal_unicode_similarity(sentences_embeddings, pair_sent_embeddings)
    uni_ratio = sum(uni_value)/len(uni_value)
    print("uni_ratio: " + str(uni_ratio))
    rev_uni_value = cal_unicode_similarity(pair_sent_embeddings, sentences_embeddings)
    rev_uni_ratio = sum(rev_uni_value)/len(rev_uni_value)
    print("rev_uni_ratio: " + str(rev_uni_ratio))
    print("avg_uni_ratio: " + str( (uni_ratio + rev_uni_ratio)/2))





parser = argparse.ArgumentParser()
parser.add_argument("-f1", "--first_file_path", type = str, 
                    help="Path to the first code file")
parser.add_argument("-f2", "--second_file_path", type=str,
                    help="Path to the second code file")

args = parser.parse_args()

Flag = True
if args.first_file_path != None: 
    if args.second_file_path != None:
        get_similarity(args.first_file_path, args.second_file_path)
        Flag = False
if Flag:
    print("Missing necessary code file paths, use --help or -h to see the detail")