from torch.optim import AdamW
import evaluate
import torch
import torch.nn.functional as F
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

        self.device = torch.device("mps")   # if torch.cuda.is_available() else torch.device("cpu")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = self.load_model(num_labels, model_name)

        self.num_epochs = num_epochs
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.optimizer = self.get_optimizer(optimizer_name)

    def get_optimizer(self, optimizer_name):
        if optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=5e-5)
        else:
            raise NotImplementedError
        return optimizer
    
    def load_model(self, num_labels, model_name="bert-base-cased"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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
            probs = F.softmax(logits, dim=-1)
            
            _, topk_indices = torch.topk(probs, k=1, dim=-1)
            predictions = topk_indices.squeeze(-1)
            # print('predictions', predictions)

            all_predictions += list(predictions.cpu().numpy())
            all_references += list(batch["labels"].cpu().numpy())
            progress_bar.update(1)

        return all_predictions, all_references
