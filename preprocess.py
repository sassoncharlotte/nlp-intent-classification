from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import List



class ProcessGoEmotions:
    positive = [
        'admiration','amusement', 'approval', 'caring',
        'desire', 'excitement', 'gratitude', 'joy',
        'love', 'optimism', 'pride', 'relief'
    ]
    negative = [
        'anger', 'annoyance', 'disappointment',
        'disapproval', 'disgust', 'embarrassment',
        'fear', 'grief', 'nervousness', 'remorse', 'sadness'
    ]
    ambiguous = [
        'confusion', 'curiosity', 'realization', 'surprise'
    ]
    neutral = [
        'neutral'
    ]

    def __init__(self, label_choice) -> None:
        self.label_choice = label_choice
        self.df = None
    
    def get_datasets(self, paths: List[str], test_size: float = 0.2):
        """ Main method """
        self.load_df(paths)
        self.encode()
        self.define_label()
        train_dataset, test_dataset = self.split_dataset(test_size)
        return train_dataset, test_dataset

    def load_df(self, paths: List[str]):
        empty = True
        for path in paths:
            df = pd.read_csv(path)
            if empty:
                self.df = df
                empty = False
            else:
                self.df = pd.concat([self.df, df], axis=1)
    
    def get_mapping(self):
        labels = ProcessGoEmotions.positive + ProcessGoEmotions.negative + \
            ProcessGoEmotions.ambiguous + ProcessGoEmotions.neutral

        mapping, mapping_category =  {}, {}
        for i, lab in enumerate(labels):
            mapping[lab] = i
            if lab in ProcessGoEmotions.positive:
                mapping_category[lab] = 3
            elif lab in ProcessGoEmotions.negative:
                mapping_category[lab] = 2
            elif lab in ProcessGoEmotions.ambiguous:
                mapping_category[lab] = 0
            elif lab in ProcessGoEmotions.neutral:
                mapping_category[lab] = 1
            else:
                print("issue")
        return labels, mapping, mapping_category

    def encode(self):
        labels, mapping, mapping_category = self.get_mapping()

        self.df["emotions"] = self.df[labels].idxmax(1)
        self.df["emotion_category"] = self.df["emotions"].replace(mapping_category)
        self.df["emotions"] = self.df["emotions"].replace(mapping)

        self.df = self.df[["text", "emotions", "emotion_category"]].copy()

    def define_label(self):
        self.df.rename({self.label_choice: "label"}, inplace=True, axis = 1)
        self.df = self.df[["text", "label"]].copy()

    def split_dataset(self, test_size: float = 0.2):
        train, test = train_test_split(self.df, test_size=test_size, random_state=42)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        train_dataset = Dataset.from_pandas(train)
        test_dataset = Dataset.from_pandas(test)

        return train_dataset, test_dataset



class TokenizeDataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def tokenize_process(self, tokenizer_name: str = "bert-base-cased"):
        self.tokenize_dataset(tokenizer_name)
        self.process_tokenized_dataset()
        return self.dataset

    def tokenize_dataset(self, tokenizer_name: str = "bert-base-cased"):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        self.dataset = self.dataset.map(tokenize_function, batched=True)

    def process_tokenized_dataset(self):
        self.dataset = self.dataset.remove_columns(["text"])
        self.dataset = self.dataset.rename_column("label", "labels")
        self.dataset.set_format("torch")
