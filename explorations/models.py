import torch
from pytorch_pretrained_bert import BertModel, BertEmbeddings

class BertEmbeddingsHT(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Apply non-linear transformation (e.g., sigmoid, tanh) to embeddings
        # to make them fat-tailed
        embeddings = torch.sigmoid(words_embeddings) + torch.tanh(position_embeddings) + torch.relu(token_type_embeddings)

        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelHT(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddingsHT(config)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        return super().forward(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)
