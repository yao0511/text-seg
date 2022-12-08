import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from pathlib2 import Path
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from utils import maybe_cuda
import sys
from argparse import ArgumentParser
import os
import accuracy
from termcolor import colored
import numpy as np


torch.multiprocessing.set_sharing_strategy('file_system')
preds_stats = utils.predictions_analysis()


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []
    
    for text, targets, path in batch:
        paths.append(path)
        
        tensor_targets = maybe_cuda(torch.LongTensor(targets))
        batched_targets.append(tensor_targets)
        
        bert_input = tokenizer.batch_encode_plus(text, pad_to_max_length=True, return_tensors='pt')
        batched_data.append(bert_input)
    
    # batched_data (sentences, {input_ids:, attention_mask:})
    # batched_targets (sentences)
        
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
        # text (sentences, )
        
        return text, targets, path
        

class Model(nn.Module):
    def __init__(self, hidden_dim, hidden_layer, batch_size):
        super().__init__()
        
        self.config = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=self.config)
        self.bert = maybe_cuda(self.bert)
        self.hidden_dim = hidden_dim
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        
        
        self.lstm = nn.LSTM(768, hidden_dim, hidden_layer, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2,2)
        
    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = maybe_cuda(d.unsqueeze(0).unsqueeze(0))
        # 進行 padding 
        padded = F.pad(v, (0,0,0, max_document_length - d_length ))  # (1, 1, max_length, 768)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 768)
    
    def forward(self, batch):
        
        batched_cls_lhs = []
        doc_len = []
        for x in batch:
            _, cls_lhs, _ = self.bert(maybe_cuda(x['input_ids']), maybe_cuda(x['attention_mask']), return_dict=False)
            print(cls_lhs.shape)
            doc_len.append(cls_lhs.shape[0])
            batched_cls_lhs.append(cls_lhs)
        max_doc_len = max(doc_len)
        padded_doc = [self.pad_document(d,max_doc_len) for d in batched_cls_lhs]
        docs_tensor = torch.cat(padded_doc, 1)
        
        x, _ = self.lstm(docs_tensor)
        
        doc_outputs = []
        for i, size in enumerate(doc_len):
            doc_outputs.append(x[0:size, i, :])
        x = torch.cat(doc_outputs, 0)
        
        x = self.linear(x)

        return x

class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for t in targets_np:
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold
      
def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if i == args.stop_after:
                break
            
            pbar.update()
            model.zero_grad()
            output = model(data)
            target_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
            loss = model.criterion(output, target_var)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)

def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target)


            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold

def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = accuracy.Accuracy()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])

                    acc.update(h, tt)

                    current_idx = to_idx

                    # acc.update(output_softmax.data.cpu().numpy(), target)

            #
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk
    
def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))
    
    dataset_path = Path(utils.config['DRCDdataset'])

    train_dataset = DRCDdataset(dataset_path / 'train')
    val_dataset = DRCDdataset(dataset_path / 'dev')
    test_dataset = DRCDdataset(dataset_path / 'trail')
    
    train_dl = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    
    model = Model(300, 1, 1)
    model.train()
    model = maybe_cuda(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_pk = 1.0
    for epoch in range(args.epochs):
        train(model, args, epoch, train_dl, logger, optimizer)
        with (checkpoint_path / 'model{:03d}.t7'.format(epoch)).open('wb') as f:
            torch.save(model, f)

        val_pk, threshold = validate(model, args, epoch, val_dl, logger)
        if val_pk < best_val_pk:
            test_pk = test(model, args, epoch, test_dl, logger, threshold)
            logger.debug(
                colored(
                    'Current best model from epoch {} with p_k {} and threshold {}'.format(j, test_pk, threshold),
                    'green'))
            best_val_pk = val_pk
            with (checkpoint_path / 'best_model.t7'.format(j)).open('wb') as f:
                torch.save(model, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=2)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
 
    main(parser.parse_args())
               
