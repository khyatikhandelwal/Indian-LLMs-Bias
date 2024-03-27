import numpy as np
import pandas as pd
import re
import ast
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub.hf_api import HfFolder

from helper_functions import Helpers
from load_model import Load_Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Obtaining Results")
    parser.add_argument("category", type=str, help="Name of the category to test.")
    parser.add_argument("model_name", type=str, help="Name of model to test.")
    parser.add_argument("instruction_prompting", type=bool, help="Whether to use instruction prompting")
    args = parser.parse_args()
    model_name = args.model_name
    category = args.category
    prompting = args.instruction_prompting

    model, tokenizer = Load_Model(model_name=model_name)()
    helpers = Helpers()

    df = helpers.load_sentences(filename=f'Data/{category}.csv',instruction_prompting=prompting)
    dict_sentences = helpers.sent_scorer(df=df, model=model, tokenizer=tokenizer)
    dict_targets = helpers.target_word_scorer(df=df, model=model, tokenizer=tokenizer)
    df['Stereotypical_Score'] = dict_sentences['Stereotypical']
    df['Antistereo_Score'] = dict_sentences['Anti-Stereotypical']
    df['Target_Stereotypical_Score'] = dict_targets['Stereotypical']
    df['Target_Antistereo_Score'] = dict_targets['Anti-Stereotypical']

    df['Score_Conditional'] = (df['Stereotypical_Score'] - df['Target_Stereotypical_Score']) - (df['Antistereo_Score']-df['Target_Antistereo_Score'])
    df['Score'] = df['Stereotypical_Score'] - df['Antistereo_Score']

    df.to_csv(f'{model_name}_{category}.csv')
    print('CLL Score:',len(df[df['Score_Conditional']>=0])/len(df))