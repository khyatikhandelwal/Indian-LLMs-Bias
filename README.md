# India-Bias Scoring

The paper can be found at: <link_to_paper>

To use decoder model scoring:

1) Clone the repository. 
2) Install all dependencies.
3) Add your huggingface token to Code/load_model.py.
```bash
python Code/decoder_model_scoring.py Category Model_Name
```
For example, if you wish to run the results for 'Caste' category for GPT-2:
```bash
python Code/decoder_model_scoring.py Caste gpt2
```

Please note that the category names need to be one of the .csv files without the '.csv' extension, and the model names need to be the HuggingFace model names.

For encoder model scoring:
Similar steps as above can be followed.
For example for Caste for mBERT: 
```bash
python Code/encoder_models_scoring.py Caste "google-bert/bert-base-multilingual-uncased"
```