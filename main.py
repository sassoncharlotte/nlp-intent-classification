from torch.utils.data import DataLoader
import seaborn as sns
import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

from preprocess import ProcessGoEmotions, TokenizeDataset
from model import TransformersModel


PATH1 = "./data/full_dataset/goemotions_1.csv"
PATH2 = "./data/full_dataset/goemotions_2.csv"
PATH3 = "./data/full_dataset/goemotions_3.csv"
PATHS = [PATH1]

# LABEL = "emotions" # all 28 labels
LABEL = "emotion_category" # positive negative ambiguous and neutral

TOKENIZER_NAME = 'roberta-base'
MODEL_NAME='roberta-base'
METRIC = "accuracy"
OPTIMIZER_NAME="AdamW"
NUM_EPOCHS = 3
BATCH_SIZE = 8
NUM_INSTANCES = 20


if __name__ == "__main__":
    # Loading train and test DataFrames
    process = ProcessGoEmotions(label_choice=LABEL)
    train, test = process.get_datasets(paths=PATHS, test_size = 0.2)

    # Converting to datasets
    train_dataset = datasets.Dataset.from_pandas(train)
    test_dataset = datasets.Dataset.from_pandas(test)

    # Tokenizing datasets
    tonekizer = TokenizeDataset(train_dataset)
    tokenized_train = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)
    tonekizer = TokenizeDataset(test_dataset)
    tokenized_test = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)

    # Subsampling datasets
    small_train_dataset = tokenized_train.shuffle(seed=42).select(range(NUM_INSTANCES))
    small_eval_dataset = tokenized_test.shuffle(seed=42).select(range(NUM_INSTANCES))

    # sample_weight = compute_sample_weight(class_weight='balanced', y=small_train_dataset["labels"])

    # Creating dataloader
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)

    # Loading model
    model = TransformersModel(
        optimizer_name=OPTIMIZER_NAME,
        num_epochs=NUM_EPOCHS,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_labels=len(tokenized_train['labels'].unique()),
        model_name=MODEL_NAME
        # sample_weights=sample_weight
    )

    # Train and test
    model.train()

    # result = model.evaluate(metric=METRIC)
    # print(f"Final {METRIC}:", result)

    # result = model.evaluate(metric=METRIC)
    y_preds, y_true = model.predict()

    if LABEL == "emotion_category":
        mapping = {0: "neutral", 1: "ambiguous", 2: "negative", 3: "positive"}

    if LABEL == "emotions":
        _, mapping, _ = process.get_mapping()
        mapping = {i: k for k, i in mapping.items()}

    report = classification_report(
        y_true, y_preds
    )

    print("\nClassification report\n")
    print(report)

    cm = confusion_matrix(y_true, y_preds, normalize='true')

    # plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, cmap='Blues')

    # set the axis labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
