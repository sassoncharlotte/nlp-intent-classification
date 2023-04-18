from torch.utils.data import DataLoader

from preprocess import ProcessGoEmotions, TokenizeDataset
from model import Model


if __name__ == "__main__":
    PATH1 = "./data/full_dataset/goemotions_1.csv"
    PATH2 = "./data/full_dataset/goemotions_2.csv"
    PATH3 = "./data/full_dataset/goemotions_3.csv"

    # LABEL = "emotions" # all 28 labels
    LABEL = "emotion_category" # positive negative ambiguous and neutral

    TOKENIZER_NAME = "bert-base-cased"
    MODEL_NAME="bert-base-cased"
    METRIC = "accuracy"
    OPTIMIZER_NAME="AdamW"
    NUM_EPOCHS = 3

    # Loading datasets
    process = ProcessGoEmotions(label_choice=LABEL)
    train_dataset, test_dataset = process.get_datasets(paths=[PATH1], test_size = 0.2)

    # Tokenizing datasets
    tonekizer = TokenizeDataset(train_dataset)
    tokenized_train = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)
    tonekizer = TokenizeDataset(test_dataset)
    tokenized_test = tonekizer.tokenize_process(tokenizer_name=TOKENIZER_NAME)

    # Subsampling datasets
    small_train_dataset = tokenized_train.shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_test.shuffle(seed=42).select(range(100))

    # Creating dataloader
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    # Loading model
    model = Model(
        optimizer_name=OPTIMIZER_NAME,
        num_epochs=NUM_EPOCHS,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_labels=len(tokenized_train['labels'].unique()),
        model_name=MODEL_NAME
    )

    # Train and test
    model.train()
    result = model.evaluate(metric=METRIC)
    print(f"Final {METRIC}:", result)
