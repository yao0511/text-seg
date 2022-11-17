import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from pathlib2 import Path
from tqdm import tqdm
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []
    
    for text, targets, path in batch:
        paths.append(path)
        
        tensor_targets = torch.LongTensor(targets)
        batched_targets.append(tensor_targets)
        
        bert_input = tokenizer.batch_encode_plus(text, pad_to_max_length=True, return_tensors='pt')
        batched_data.append(bert_input)
        
    return batched_data, batched_targets, paths
        

class DRCDdataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.files = list(Path(data_path).glob('*.txt'))
    
    def __getitem__(self, index):
        path = self.files[index]
        
        return self.read_DRCD_file(path)
    
    def __len__(self):
        return len(self.files)
    
    def read_DRCD_file(self, path):
        seperator = '=========='
        with Path(path).open('r', encoding='utf-8') as f:
            raw_text = f.read()
        paragraphs = [p for p in raw_text.strip().split(seperator) if len(p)>2]
        
        targets = []
        text = []
        for paragraph in paragraphs:
            sentences = [s for s in paragraph.split('\n') if len(s.split()) > 0]
            sentences_targets = [0 for s in sentences[:-1]]
            sentences_targets.append(1)
            targets.extend(sentences_targets)
        
        
            for sentence in sentences:
                text.append(sentence)
        
        return text, targets, path
        

class Model(nn.Module):
    def __init__(self, hidden_dim, hidden_layer, batch_size):
        super().__init__()
        
        self.config = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=self.config)
        self.hidden_dim = hidden_dim
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        
        
        self.lstm = nn.LSTM(768, hidden_dim, hidden_layer, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2,2)
        
    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0,0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)
    
    def forward(self, batch):
        
        batched_cls_lhs = []
        doc_len = []
        for x in batch:
            _, cls_lhs, _ = self.bert(x['input_ids'], x['attention_mask'], return_dict=False)
            doc_len.append(cls_lhs.shape[0])
            batched_cls_lhs.append(cls_lhs)
        max_doc_len = max(doc_len)
        padded_doc = [self.pad_document(d,max_doc_len) for d in batched_cls_lhs]
        docs_tensor = torch.cat(padded_doc, 1)
        
        x, _ = self.lstm(docs_tensor)
        x = self.linear(x)

        return x
        

train_dataset = DRCDdataset(data_path=r'C:\Users\vince_wang\research\evaluate\8.24\DRCD\train')
train_dl = DataLoader(train_dataset, batch_size=3, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(train_dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)
model = Model(300,1,1)

with tqdm(desc='Testing', total=len(val_dl)) as pbar:
    model.eval()
    for i, (data, targets, path) in enumerate(val_dl):
        print(i)
        out = model(data)
        print(out.shape)
        pbar.update()