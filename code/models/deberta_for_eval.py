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

        # self.regression_heads = torch.nn.ModuleList([torchvision.ops.MLP(1536, [512, 1], dropout=0.05) for _ in range(n_supervision)]) Somehow this breaks the computation graph when used in conjunction with torch.cat
        self.regression_heads_layer_1 = torch.nn.ModuleList([torch.nn.Linear(768, 512) for i in range(n_supervision)])
        self.regression_heads_layer_2 = torch.nn.ModuleList([torch.nn.Linear(512, 1) for i in range(n_supervision)])
        self.relu = torch.nn.ReLU()

        self.to(device)
        self.float()
    
    def forward(self, sents):
        tokenized = self.tokenizer(sents, padding=True, truncation=True, max_length=512)
        input_ids = torch.tensor(tokenized['input_ids']).to(self.device)
        token_type_ids =  torch.tensor(tokenized['token_type_ids']).to(self.device)
        attention_mask =  torch.tensor(tokenized['attention_mask']).to(self.device)
        model_out = self.deberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :] # Take the emb for the first token

        heads_out = []
        for head_idx in range(self.n_supervision):
            head_out = self.regression_heads_layer_1[head_idx](model_out)
            head_out = self.relu(head_out)
            head_out = self.regression_heads_layer_2[head_idx](head_out)
            heads_out.append(head_out)

        heads_out = torch.cat(heads_out, dim=1)
        return heads_out # [batch_size, n_head]