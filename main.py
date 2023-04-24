import torch
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils import compute_class_weight


from preprocess import ProcessGoEmotions, TokenizeDataset #, WeightedDataset
from model import TransformersModel


PATH1 = "./data/full_dataset/goemotions_1.csv"
PATH2 = "./data/full_dataset/goemotions_2.csv"
PATH3 = "./data/full_dataset/goemotions_3.csv"
PATHS = [PATH1]

LABEL = "emotions"  # all 28 labels
# LABEL = "emotion_category"  # positive, negative, ambiguous and neutral

TOKENIZER_NAME = 'roberta-base'
MODEL_NAME='roberta-base'
METRIC = "accuracy"
OPTIMIZER_NAME="AdamW"
NUM_EPOCHS = 1
BATCH_SIZE = 16
NUM_INSTANCES = 800

# Class imbalance-addressing methods (change *ONE* of these values to True to implement the corresponding approach)
CLASS_WEIGHTING = False
WEIGHTED_RANDOM_SAMPLING = False

if __name__ == "__main__":
    # Loading train and test DataFrames
    process = ProcessGoEmotions(label_choice=LABEL)
    train, test = process.get_datasets(paths=PATHS, test_size = 0.2, drop_neutral=True)

    # print('unique labels in train.label: \n', sorted(train['label'].unique().tolist()))
    # class_counts = train['label'].value_counts()
    # print('class_counts: \n', class_counts)

    # Converting to datasets
    train_dataset = datasets.Dataset.from_pandas(train)
    test_dataset = datasets.Dataset.from_pandas(test)

    # Tokenizing datasets
    tonekizer = TokenizeDataset(train_dataset)
    tokenized_train = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)
    tonekizer = TokenizeDataset(test_dataset)
    tokenized_test = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)

    # Subsampling datasets (IMPORTANT: make sure all classes are represented in these subsampled datasets)
    small_train_dataset = tokenized_train.shuffle(seed=42).select(range(NUM_INSTANCES))
    small_eval_dataset = tokenized_test.shuffle(seed=42).select(range(NUM_INSTANCES))
    # small_train_dataset = tokenized_train
    # small_eval_dataset = tokenized_test

    if CLASS_WEIGHTING or WEIGHTED_RANDOM_SAMPLING:

        # Computing class weights
        class_weights = compute_class_weight('balanced', classes=train['label'].unique(), y=train['label'])
        class_weights = torch.from_numpy(class_weights).float()

        # Creating sampler
        sampler = WeightedRandomSampler(
            weights=class_weights, num_samples=len(small_train_dataset), replacement=True
        )

    # Creating dataloader
    if WEIGHTED_RANDOM_SAMPLING:
        train_dataloader = DataLoader(small_train_dataset, sampler=sampler, batch_size=BATCH_SIZE)  # if training WITH weighted random sampling
    else:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)  # if training WITHOUT weighted random sampling
    
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)

    # Loading model
    model = TransformersModel(
        optimizer_name=OPTIMIZER_NAME,
        num_epochs=NUM_EPOCHS,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_labels=len(tokenized_train['labels'].unique()),
        model_name=MODEL_NAME
    )

    # Training
    if CLASS_WEIGHTING:
        model.train(class_weights=class_weights)  # if training WITH class weighting
    else:
        model.train()  # if training WITHOUT class weighting

    # Evaluating the model
    result = model.evaluate(metric=METRIC)
    print(f"Final {METRIC}:", result)
    
    # Computing the classification report and the confusion matrix
    y_preds, y_true = model.predict()

    if LABEL == "emotion_category":
        mapping = {0: "neutral", 1: "ambiguous", 2: "negative", 3: "positive"}

    if LABEL == "emotions":
        _, mapping, _ = process.get_mapping()
        mapping = {i: k for k, i in mapping.items()}

    report = classification_report(
        y_true, y_preds
    )

    print('unique y_true', np.unique(y_true))
    print('unique y_preds', np.unique(y_preds))

    print("\nClassification report\n")
    print(report)

    cm = confusion_matrix(y_true, y_preds, normalize='true')

    # Plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, cmap='Blues')

    # Set the axis labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
