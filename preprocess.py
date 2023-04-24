from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import List

# import random
# import pandas as pd

# def undersample(df, target_col):
#     # get class counts
#     class_counts = df[target_col].value_counts()
    
#     # get size of minority class
#     min_class_size = class_counts.min()
    
#     # create empty dataframe for undersampled data
#     undersampled_df = pd.DataFrame(columns=df.columns)
    
#     # for each class
#     for class_label in class_counts.index:
#         # get instances of the class
#         class_df = df[df[target_col] == class_label]
        
#         # randomly select instances to keep
#         keep_indices = random.sample(range(len(class_df)), min_class_size)
#         keep_df = class_df.iloc[keep_indices, :]
        
#         # add undersampled instances to the new dataframe
#         undersampled_df = undersampled_df.append(keep_df)
        
#    return undersampled_df

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
    
    def get_datasets(self, paths: List[str], test_size: float = 0.2, drop_neutral: bool = True):
        """ Main method """
        self.load_df(paths)
        self.encode()
        self.define_label()
        train_dataset, test_dataset = self.split_dataset(test_size, drop_neutral)
        return train_dataset, test_dataset

    def load_df(self, paths: List[str]):
        empty = True
        for path in paths:
            df = pd.read_csv(path)
            if empty:
                self.df = df
                empty = False
            else:
                self.df = pd.concat([self.df, df], axis=0)

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
        assert self.label_choice in self.df.columns, "Label choice must be 'emotions' or 'emotion_category'."
    
        self.df.rename({self.label_choice: "label"}, inplace=True, axis = 1)
        self.df = self.df[["text", "label"]]

    def split_dataset(self, test_size: float = 0.2, drop_neutral: bool = True):
        _, mapping, _ = self.get_mapping()
        if drop_neutral:
            print("Removing 'neutral'")
            self.df = self.df[self.df.label != mapping['neutral']]
        
        # self.df = undersample(self.df, 'label')
        
        train, test = train_test_split(self.df, test_size=test_size, stratify=self.df.label, random_state=42)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        return train, test





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
