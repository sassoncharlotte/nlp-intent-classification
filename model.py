from torch.optim import AdamW
import evaluate
import torch
from transformers import AutoModelForSequenceClassification, get_scheduler, RobertaModel
from tqdm.auto import tqdm



class TransformersModel:
    def __init__(
            self,
            optimizer_name,
            num_epochs,
            train_dataloader,
            eval_dataloader,
            num_labels,
            model_name = "bert-base-cased"
        ) -> None:

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = self.load_model(num_labels, model_name)

        self.num_epochs = num_epochs
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.optimizer = self.get_optimizer(optimizer_name)
        self.lr_scheduler = self.get_scheduler()


    def get_optimizer(self, optimizer_name):
        if optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=1e-3)
        else:
            raise NotImplementedError
        return optimizer
    

    def get_scheduler(self):
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )
        return lr_scheduler
    
    def load_model(self, num_labels, model_name="bert-base-cased"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # model = RobertaModel.from_pretrained(model_name, num_labels=num_labels)

        # Freeze all layers except for the last one
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Only update the last layer
                param.requires_grad = False

        model.to(self.device)
        return model

    def train(self, class_weights=None):
        progress_bar = tqdm(range(self.num_training_steps))

        # Training
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # To perform class weighting
                if class_weights is not None:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss = criterion(outputs.logits, batch["labels"].long().to(self.device))
                else:
                    loss = outputs.loss
                
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_postfix({"loss": loss.item()})
                progress_bar.update(1)


    def evaluate(self, metric = "accuracy"):
        metric = evaluate.load(metric)
        self.model.eval()
        progress_bar = tqdm(range(len(self.eval_dataloader)))
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            # progress_bar.set_postfix({f"{metric}": metric.compute()})
            progress_bar.update(1)

        return metric.compute()


    def predict(self):
        all_predictions, all_references = [], []
        self.model.eval()
        progress_bar = tqdm(range(len(self.eval_dataloader)))
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions += list(predictions.numpy())
            all_references += list(batch["labels"].numpy())
            progress_bar.update(1)

        return all_predictions, all_references
