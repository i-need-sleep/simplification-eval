import torch
import torchvision
from transformers import AutoModel, AutoTokenizer

class DebertaForEval(torch.nn.Module):
    def __init__(self, model_path, tokenizer_path, device, n_supervision=13):
        super(DebertaForEval, self).__init__()
        self.n_supervision = n_supervision
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.deberta = AutoModel.from_pretrained(model_path)

        self.regression_heads = torch.nn.ModuleList([torchvision.ops.MLP(1536, [512, 1], dropout=0.05) for _ in range(n_supervision)])
        self.to(device)
    
    def forward(self, sents):
        tokenized = self.tokenizer(sents, padding=True, truncation=True)
        input_ids = torch.tensor(tokenized['input_ids']).to(self.device)
        token_type_ids =  torch.tensor(tokenized['token_type_ids']).to(self.device)
        attention_mask =  torch.tensor(tokenized['attention_mask']).to(self.device)
        model_out = self.deberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :] # Take the emb for the first token
        heads_out = [head(model_out) for head in self.regression_heads]
        heads_out = torch.cat(heads_out, dim=1)
        return heads_out # [batch_size, n_head]