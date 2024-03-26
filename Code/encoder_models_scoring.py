# This code has been taken from https://github.com/kanekomasahiro/evaluate_bias_in_mlm
# We calculate the AUL Scores using this code

import json
import argparse
import torch
import difflib
import nltk
import re
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, BertForMaskedLM

def fill_masks(sentence, targets):
        new_sentence = sentence
        for target in targets:
            new_sentence = re.sub('MASK',target,new_sentence, count=1)
        #print("FILLED MASK:", new_sentence)
        return new_sentence

def read_file(filename):
    df = pd.read_csv(filename)
    df['Target_Stereotypical'] = df['Target_Stereotypical'].apply(lambda x: x.replace("'", "").replace('[', '').replace(']','').split(','))
    df['Target_Anti-Stereotypical'] = df['Target_Anti-Stereotypical'].apply(lambda x: x.replace("'", "").replace('[', '').replace(']','').split(','))
    df['Stereotypical'] = df.apply(lambda x: fill_masks(x['Sentence'],x['Target_Stereotypical']),axis=1)
    df['Anti-Stereotypical'] = df.apply(lambda x: fill_masks(x['Sentence'],x['Target_Anti-Stereotypical']),axis=1)
    return df

def calculate_aul(model, sentence, log_softmax, attention=False):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    tokens = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

    # Get the token IDs and attention mask
    input_ids = tokens['input_ids']
    output = model(input_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    input_ids = input_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, input_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    hidden_states = output.hidden_states[-1][:,1:-1]
    hidden_state = torch.mean(hidden_states, 1).detach().numpy()

    return score, hidden_state

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Obtaining Results")
    parser.add_argument("category", type=str, help="Name of the category to test.")
    parser.add_argument("model_name", type=str, help="Name of model to test.")
    args = parser.parse_args()
    model_name = args.model_name
    category = args.category

    df = read_file(filename=f"Data/{category}.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name,output_hidden_states=True, output_attentions=True)

    mask_id = tokenizer.mask_token_id
    log_softmax = torch.nn.LogSoftmax(dim=1)

    stereo_inputs = [i for i in df['Stereotypical']]
    antistereo_inputs = [i for i in df['Anti-Stereotypical']]

    stereo_scores = []
    antistereo_scores = []
    stereo_embes = []
    antistereo_embes = []

    for i in stereo_inputs:
        stereo_score, stereo_hidden_state = calculate_aul(model, i, log_softmax, attention=False)
        stereo_scores.append(stereo_score)
        stereo_embes.append(stereo_hidden_state)

    for j in antistereo_inputs:
        antistereo_score, antistereo_hidden_state = calculate_aul(model, j, log_softmax, attention=False)
        antistereo_scores.append(antistereo_score)
        antistereo_embes.append(antistereo_hidden_state)

    stereo_scores = np.array(stereo_scores)
    stereo_scores = stereo_scores.reshape([-1, 1])
    antistereo_scores = np.array(antistereo_scores)
    antistereo_scores = antistereo_scores.reshape([1, -1])
    bias_scores = stereo_scores > antistereo_scores

    stereo_embes = np.concatenate(stereo_embes)
    antistereo_embes = np.concatenate(antistereo_embes)
    weights = cos_sim(stereo_embes, antistereo_embes.T)

    weighted_bias_scores = bias_scores * weights
    bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
    print('bias score (emb):', round(bias_score * 100, 2))