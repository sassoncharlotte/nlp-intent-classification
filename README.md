# ANLP Project - Fine-Grained Emotion Classification

In this project, we compare different methods presented in our paper *Addressing Class Imbalance in Fine-Grained Emotion Classification: A Comparative Study*, aimed at improving a BERT-based model's performance on fine-grained emotion classification tasks.

## 🏁 Getting started

Command to retrieve the data (3 CSV files should be in a folder ```full_dataset```, itself in the ```data``` folder, at the root of the repository):

```bash
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

Create a ```Python``` environment (preferably running Python 3.9) and activate it using the following commands.
```bash
conda create -n mynlp python=3.9
```
```bash
conda activate
```

To access the directory, use:
```bash
cd nlp-intent-classification
```

You may then install the requirements using the command:
```
pip install -r requirements.txt
```

Finally, run the file ```main.py``` using:

Then, run the file main.py with the arguments of your choice.

## Structure of the repository

- ```.gitignore```: used for Git to ignore some files
- ```data_analysis.py```: analysis of the dataset
- ```main.py```: file to run the code
- ```model.py```: contains TransformersModel class used to load, train and evaluate the model
- ```preprocess.py```: contains ProcessGoEmotions and TokenizeDataset classes, used to preprocess the data
- ```README.md```: README file
- ```requirements.txt```: contains the package requirements and dependencies
