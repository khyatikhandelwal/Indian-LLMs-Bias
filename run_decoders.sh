#!/bin/bash

# Run the Python script

for m in gpt2 meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf tiiuae/falcon-7b-instruct tiiuae/falcon-40b-instruct
do 
    echo $m
    for f in Caste India_Religious Race Gender
    do 
        echo $f
        python Code/decoder_model_scoring.py $f $m true
    done
done
