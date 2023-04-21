# NLP - Fine-Grained Emotion Classification

Commande pour télécharger les données :

```bash
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

## 🏁 Getting started

Create a Python environment and activate it. After that, you need to use the command:
```
pip install -r requirements.txt
```

Then, run the file main.py with the arguments of your choice.

## Structure of the repository

- .gitignore: used for Git to ignore some files
- data_analysis.py: Analysis of the dataset
- main.py: file to run the code
- model.py: contains TransformersModel class used to load, train and evaluate the model
- preprocess.py: contains ProcessGoEmotions and TokenizeDataset classes, used to preprocess the data
- README.md: README file
- requirements.txt: contains the package requirements
