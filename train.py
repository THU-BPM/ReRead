import pdb
import re
import sys
import torch.optim as optim
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import os
import copy
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_recall_fscore_support
from collections import Counter
import csv
import argparse

def setRandomSeed():
    # Set the seed value all over the place to make this reproducible.
    # The uploaded pretrained BERT model is trained with random seed 42.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
setRandomSeed()

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Bert Classification For CHEF')
parser.add_argument('--noload', action='store_true', help='if present, do not load any saved model')
parser.add_argument('--cuda', type=str, default="0",help='appoint GPU devices')
parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--initial_lr', type=float, default=3e-5, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=4, help='training epochs for labeled data')
parser.add_argument('--total_epochs', type=int, default=10, help='total epochs of the RL learning')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = torch.device("cuda")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    # if len(labels_flat[non_zero_idx])==0:
    #     print("error occur: ", labels_flat)
    #     return 0
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pre_processing(sentence_train, sentence_train_label, domains, bert_type):
    input_ids = []
    attention_masks = []
    labels = []
    # domains = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(sentence_train_label[i])
        # index_list.append(i)
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    domains = torch.tensor(domains, device='cuda')
    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, domains)
    return train_dataset, tokenizer


def pre_processing_evidence_extractor(sentence_train, gold_evidence, labels, domains):
    input_ids = []
    attention_masks = []
    gold_evidence_mask = []
    # domains = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        # remove everyting in [CLS] ... [SEP] pair with regex, shortest match
        gold_evidence_only = gold_evidence[i].split(' [SEP] ')[:-1]
        sentence_evidence_only = re.sub(r' \[CLS\].*?\[SEP\] ', '', sentence_train[i]).split(' [SEP] ')[:-1]
        # remove space
        gold_evidence_only = [s.replace(' ', '') for s in gold_evidence_only]
        sentence_evidence_only = [s.replace(' ', '') for s in sentence_evidence_only]
        is_gold_evidence = [1 if sentence_evidence_only[i] in gold_evidence_only else 0 for i in range(len(sentence_evidence_only))]
        is_gold_evidence += [0] * (25 - len(is_gold_evidence))
        gold_evidence_mask.append(is_gold_evidence)
        
        # index_list.append(i)
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    gold_evidence_mask = torch.tensor(gold_evidence_mask).to(device)
    labels = torch.tensor(labels, device='cuda')
    domains = torch.tensor(domains).to(device)
    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, gold_evidence_mask, labels, domains)
    return train_dataset, tokenizer




def stratified_sample(dataset, ratio):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        sampled_indices += indices[0:int(len(indices) * ratio)]
        rest_indices += indices[int(len(indices) * ratio):len(indices)]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]

def drop1(dataset, max1num):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    indices = data_dict[1]
    random.shuffle(indices)
    sampled_indices = []
    sampled_indices += indices[0:max1num]
    sampled_indices += (data_dict[0] + data_dict[2])
    return Subset(dataset, sampled_indices)


def prepareToTrainBertForSequenceClassification(sentence, sentence_label, domains, bert_type, model_save_dir):
    dataset, tokenizer = pre_processing(sentence, sentence_label, domains, bert_type)
    # split train and validation dataset
    val_dataset = Subset(dataset, [i for i in range(999)])
    train_dataset = Subset(dataset, [i for i in range(999, len(dataset))])
    # train_dataset, val_dataset = stratified_sample(dataset, 0.8)
    # train_len = int(len(dataset) * 0.8)
    # train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset)-train_len])
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size
    )

    # Load models
    model = BertForSequenceClassification.from_pretrained(
        bert_type,      # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--3
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model = nn.DataParallel(model)
    model = model.to(device)
    return model, train_dataloader, val_dataloader, bert_type, model_save_dir

class EvidenceExtractor(nn.Module):
    def __init__(self):
        super(EvidenceExtractor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(256, 50)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

def prepareToTrain(sentence, gold_evidence, labels, domains):
    dataset, tokenizer = pre_processing_evidence_extractor(sentence, gold_evidence, labels, domains)
    # split train and validation dataset
    val_dataset = Subset(dataset, [i for i in range(999)])
    train_dataset = Subset(dataset, [i for i in range(999, len(dataset))])
    # train_dataset, val_dataset = stratified_sample(dataset, 0.8)
    # train_len = int(len(dataset) * 0.8)
    # train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset)-train_len])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size
    )

    # Load models
    model = EvidenceExtractor()

    model = nn.DataParallel(model)
    model = model.to(device)

    
    # Load models
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',      # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--3
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    bert_model = nn.DataParallel(bert_model)
    bert_model = bert_model.to(device)
    return bert_model, model, train_dataloader, val_dataloader




def format_evidence(evidence):
    full_text_list = [item['text'] for item in evidence.values()]
    # to one str
    full_text = ''.join(full_text_list)
    # tokenize sentence
    full_text_list = re.split(r'[？：。！（）.“”…\t\n]', full_text)
    full_text_list = [item for item in full_text_list if len(item) > 5][:25]
    return ' [SEP] '.join(full_text_list) + ' [SEP] '

def get_claim_evidence_sentence():
    datalist = json.load(open('./data/CHEF_test.json', 'r', encoding='utf-8')) \
        + json.load(open('./data/CHEF_train.json', 'r', encoding='utf-8'))
    claims = [' [CLS] ' + row['claim']  + ' [SEP] ' for row in datalist]
    evidences = [format_evidence(row['evidence']) for row in datalist]
    sentence = [claim + evidence for claim, evidence in zip(claims, evidences)]
    return sentence

def get_gold_evidence_sentence():
    datalist = json.load(open('./data/CHEF_test.json', 'r', encoding='utf-8')) \
        + json.load(open('./data/CHEF_train.json', 'r', encoding='utf-8'))
    gold_evidences = [format_evidence(row['gold evidence']) for row in datalist]
    return gold_evidences

def main(argv = None):
    datalist = json.load(open('./data/CHEF_test.json', 'r', encoding='utf-8')) \
        + json.load(open('./data/CHEF_train.json', 'r', encoding='utf-8'))
    labels = [row['label'] for row in datalist]
    domain2id = {
        '政治':0, '社会':1, '生活':1, '文化':2, '科学':3, '公卫':4
    }
    """
    # count f1 according to different length
    domains = []
    for row in datalist:
        ids = len(row['claim']) // 10
        if ids > 3:
            ids = 3
        domains.append(ids)
    """
    domains = [domain2id[row['domain']] for row in datalist]
    print('====================Init model and dataset...=================')
    sentence = get_claim_evidence_sentence()
    gold_evidence = get_gold_evidence_sentence()
    bert_model, extractor_model, train_dataloader, val_dataloader = \
        prepareToTrain(
            sentence, gold_evidence, labels, domains
        )
    print('===================Init model and dataset done=================')
    setRandomSeed()
    print('===========BertForSequenceClassification train begin===========')
    best_microf1_cls, best_precision_cls, best_recall_cls, best_macrof1_cls = \
        train_and_save_bert_model(bert_model, train_dataloader, val_dataloader, 'bert-base-chinese', './bert/', no_load=args.noload)
    print('============BertForSequenceClassification train end============')
    setRandomSeed()
    print('================Evidence Extraction train begin================')
    # (input_ids, attention_masks, gold_evidence_mask, labels, domains)
    best_microf1_ext, best_precision_ext, best_recall_ext, best_macrof1_ext = \
        train_and_save_extractor_model(extractor_model, bert_model, train_dataloader, val_dataloader, no_load=args.noload)
    setRandomSeed()
    print('=================Evidence Extraction train end=================')
    print('===================BertForSequenceClassification===============')
    print("       F1 (micro): {:.3%}".format(best_microf1_cls))
    print("Precision (macro): {:.3%}".format(best_precision_cls))
    print("   Recall (macro): {:.3%}".format(best_recall_cls))
    print("       F1 (macro): {:.3%}".format(best_macrof1_cls))
    print('===============================================================')
    print('=========================Evidence Extractor====================')
    print("       F1 (micro): {:.3%}".format(best_microf1_ext))
    print("Precision (macro): {:.3%}".format(best_precision_ext))
    print("   Recall (macro): {:.3%}".format(best_recall_ext))
    print("       F1 (macro): {:.3%}".format(best_macrof1_ext))
    print('===============================================================')
    
def train_and_save_bert_model(model, train_dataloader, val_dataloader, bert_type='bert', model_save_dir='model_save', no_load=False):
    # define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )

    total_steps = len(train_dataloader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    training_stats = []
    total_t0 = time.time()

    param_x = []
    param_y = []
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    best_prediction = None
    best_ground_truth = None

    # for step, batch in enumerate(val_dataloader):
    #     b_labels = batch[2].to(device)
    #     # if all zero print "all zero"
    #     if torch.sum(b_labels) == 0:
    #         pass
    #     else:
    #         import pdb
    #         print('not all zero')
    #         pdb.set_trace()

    if no_load == False:
        # try to load model
        try:
            # 'bertforseqcls.bin'
            model.load_state_dict(torch.load('./bertforseqcls.bin'))
            print('load model')
        except:
            print('no model to load')
        
        if input("Train BertForSequenceClassification? [yes] / no: ") == 'no':
            return -1, -1, -1, -1
    
    for epoch_i in range(0, args.epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode.
        model.train()

        # save mixup features
        all_logits = np.array([])
        all_ground_truth = np.array([])

        # For each batch of training data...
        epoch_params = []
        for step, batch in enumerate(train_dataloader):
            # break
            # (input_ids, attention_masks, gold_evidence_mask, labels, domains)
            batch_params = np.array([])
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed), end=' ')
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[3].to(device)
            model.zero_grad()


            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask, 
                labels=b_labels
            )
            loss, logits = outputs[0], outputs[1]
            # loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
            print(f'loss: {loss.sum().item()}')
            total_train_loss += loss.sum().item()
            # Perform a backward to calculate the gradients.
            loss.sum().backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            labels_flat = label_ids.flatten()
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)
        
        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()
        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])

        """
        print('\ntrain data score:')
        for batch in train_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[3].to(device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss, logits = outputs[0], outputs[1]
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.3%}".format(f1))
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.3%}".format(pre))
        print("   Recall (macro): {:.3%}".format(recall))
        print("       F1 (macro): {:.3%}".format(f1))
        """
        print("")
        print("Running Validation...")
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])
        all_domains = np.array([])
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[3].to(device)
            b_domains = batch[4].to('cpu').numpy()
            with torch.no_grad():
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss, logits = outputs[0], outputs[1]
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            all_domains = np.concatenate((all_domains, b_domains), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print('Validation Elapsed: {:}.'.format(validation_time))
        # print(bert_type)
        c = Counter()
        for pred in all_prediction:
            c[int(pred)] += 1
        print(c)
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.2%}".format(f1))
        microf1 = f1
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.2%}".format(pre))
        print("   Recall (macro): {:.2%}".format(recall))
        print("       F1 (macro): {:.2%}".format(f1))
        if f1 > best_macrof1:
            best_microf1 = microf1
            best_macrof1 = f1
            best_recall = recall
            best_precision = pre
            print('Above is best')
            # display every label's f1 score
            pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction)
            print("Precision:", pre)
            print("   Recall:", recall)
            print("       F1:", f1)
            # calculate every domain's f1 score
            all_domains = np.around(all_domains).astype(int)
            # there are 5 domains
            domains_num = 5
            ground_truths = [[] for i in range(domains_num)]
            predictions = [[] for i in range(domains_num)]
            for i in range(len(all_ground_truth)):
                ground_truths[all_domains[i]].append(all_ground_truth[i])
                predictions[all_domains[i]].append(all_prediction[i])
            for i in range(domains_num):
                print('length id', i, 'score: ')
                pre, recall, f1, _ = precision_recall_fscore_support(ground_truths[i], predictions[i], average='micro')
                print("       F1 (micro): {:.2%}".format(f1))
                pre, recall, f1, _ = precision_recall_fscore_support(ground_truths[i], predictions[i], average='macro')
                print("Precision (macro): {:.2%}".format(pre))
                print("   Recall (macro): {:.2%}".format(recall))
                print("       F1 (macro): {:.2%}".format(f1))
                print('================' * 3)

        # break
    # save model as "bertforseqcls.bin"
    print("****************************")
    print("Best Validation Record:")
    print("       F1 (micro): {:.2%}".format(best_microf1))
    print("Precision (macro): {:.2%}".format(best_precision))
    print("   Recall (macro): {:.2%}".format(best_recall))
    print("       F1 (macro): {:.2%}".format(best_macrof1))
    print("****************************")

    torch.save(model.state_dict(), './bertforseqcls.bin')
    return best_microf1, best_precision, best_recall, best_macrof1


def train_and_save_extractor_model(extractor_model, bert_model, train_dataloader, val_dataloader, bert_type='bert', model_save_dir='model_save', no_load=False):
    # define the loss function
    # we need to copy another bert_model
    bert_finetuned = copy.deepcopy(bert_model)

    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese', # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.  
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    ).to(bert_finetuned.module.device)

    # weights = torch.tensor([1.0, 2.0]).to(device)
    weights = torch.tensor([1.0, 10.0]).to(device)
    # because gold evidence << all evidence, resulting in 
    # overwhemling negative samples for the evidence extractor.
    criterion = nn.CrossEntropyLoss(weight=weights)
    extractor_epochs = 4

    optimizer = AdamW(
        list(extractor_model.parameters()) + list(bert_model.parameters()),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps,  # args.adam_epsilon  - default is 1e-8.,
        weight_decay=0.01
    )

    total_steps = len(train_dataloader) * extractor_epochs * 2 * 2

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    training_stats = []
    total_t0 = time.time()

    param_x = []
    param_y = []
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    best_prediction = None
    best_ground_truth = None

    if no_load == False:
        # try to load model
        try:
            # 'bertforseqcls.bin'
            extractor_model.load_state_dict(torch.load('./evidenceextractor.bin'))
            bert_model.load_state_dict(torch.load('./bertforseqcls_for_ext.bin'))
            print('load models')
        except:
            print('no model to load')
        
        if input("Train Evidence Extractor? [yes] / no: ") == 'no':
            return -1, -1, -1, -1

    for epoch_i in range(0, extractor_epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, extractor_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode.
        extractor_model.train()
        bert_model.train()

        # save mixup features
        all_logits = np.array([])
        all_ground_truth = np.array([])

        # For each batch of training data...
        epoch_params = []
        for step, batch in enumerate(train_dataloader):
            batch_params = np.array([])
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed), end=' ')
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_01 = batch[2].to(device)
            b_labels = batch[3].to(device)

            extractor_model.train()
            bert_model.train()
            extractor_model.zero_grad()
            bert_model.zero_grad()

            outputs_bert = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask, 
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            # [16, 25, 2]
            # softmax on last dim
            outputs = F.softmax(outputs, dim=-1)

            # cross entropy loss
            loss_plau = criterion(outputs.view(-1, 2), b_labels_01.view(-1)) / 100
            # loss_plau = 0
            loss_plau.sum().backward()
            torch.nn.utils.clip_grad_norm_(extractor_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            # optimizer.step()
            # scheduler.step()

            extractor_model.zero_grad()
            bert_model.zero_grad()
            outputs_bert = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask, 
                output_hidden_states=True
            )
            cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
            # torch.cuda.empty_cache()
            
            b_input_mask_claim, b_input_mask_25_evidence = get_input_masks_for_claim_and_evidence(b_input_mask, b_input_ids)            

            outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
            outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
            # outputs = F.softmax(outputs, dim=-1)[:, :, -1]
            outputs_inv = torch.ones_like(outputs) - outputs
            
            b_input_mask_use_extractor = torch.bmm(outputs.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim
            b_input_mask_use_extractor_inv = torch.bmm(outputs_inv.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim

            outputs_use_extractor = bert_finetuned(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask_use_extractor, 
                labels=b_labels
            )
            loss_use_extractor, logits = outputs_use_extractor[0], outputs_use_extractor[1]
            # torch.cuda.empty_cache()

            outputs_use_extractor_inv = bert_finetuned(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask_use_extractor_inv, 
                labels=b_labels
            )
            loss_use_extractor_inv = outputs_use_extractor_inv[0]
            # torch.cuda.empty_cache()
            loss_full_suff = loss_use_extractor - loss_use_extractor_inv
            loss_full_suff.sum().backward()
            torch.nn.utils.clip_grad_norm_(extractor_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss = loss_use_extractor - loss_use_extractor_inv + loss_plau
            
            print(f'loss: {loss.sum().item():4f} = {loss_use_extractor.sum().item():4f} - {loss_use_extractor_inv.sum().item():4f} + {loss_plau:4f}', end='\r')
            total_train_loss += loss.sum().item()
            # Perform a backward to calculate the gradients.
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            labels_flat = label_ids.flatten()
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)
        
        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()
        # Put the model in evaluation mode
        extractor_model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])

        """
        print('\ntrain data score:')
        for idx, batch in enumerate(train_dataloader):
            print(f'batch {idx:4d}', end='\r')
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_01 = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            with torch.no_grad():
                outputs_bert = bert_model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask, 
                    output_hidden_states=True
                )
                cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
                # torch.cuda.empty_cache()
                outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
                # [16, 25, 2]
                # softmax on last dim
                outputs = F.softmax(outputs, dim=-1)

                # cross entropy loss
                loss_plau = criterion(outputs.view(-1, 2), b_labels_01.view(-1))
                
                b_input_mask_claim, b_input_mask_25_evidence = get_input_masks_for_claim_and_evidence(b_input_mask, b_input_ids)            

                outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
                outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
                # outputs = F.softmax(outputs, dim=-1)[:, :, -1]
                outputs_inv = torch.ones_like(outputs) - outputs
                
                b_input_mask_use_extractor = torch.bmm(outputs.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim
                b_input_mask_use_extractor_inv = torch.bmm(outputs_inv.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim

                outputs_use_extractor = bert_finetuned(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask_use_extractor, 
                    labels=b_labels
                )
                loss_use_extractor, logits = outputs_use_extractor[0], outputs_use_extractor[1]
                # torch.cuda.empty_cache()

                outputs_use_extractor_inv = bert_finetuned(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask_use_extractor_inv, 
                    labels=b_labels
                )
                loss_use_extractor_inv = outputs_use_extractor_inv[0]
                # torch.cuda.empty_cache()
                
                loss = loss_use_extractor - loss_use_extractor_inv + loss_plau

            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.3%}".format(f1))
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.3%}".format(pre))
        print("   Recall (macro): {:.3%}".format(recall))
        print("       F1 (macro): {:.3%}".format(f1))
        """
        print("")
        print("Running Validation...")
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])
        all_domains = np.array([])
        # Evaluate data for one epoch
        for idx, batch in enumerate(val_dataloader):
            print(f'validation batch {idx:4d} / {len(val_dataloader):4d}', end='\r')
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_01 = batch[2].to(device)
            b_labels = batch[3].to(device)
            b_domains = batch[4].to('cpu').numpy()
            
            with torch.no_grad():
                outputs_bert = bert_model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask, 
                    output_hidden_states=True
                )
                cls_last_hidden_state = outputs_bert[1][-1][:, 0, :]
                # torch.cuda.empty_cache()
                outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
                # [16, 25, 2]
                # softmax on last dim
                outputs = F.softmax(outputs, dim=-1)

                # cross entropy loss
                loss_plau = criterion(outputs.view(-1, 2), b_labels_01.view(-1))
                
                b_input_mask_claim, b_input_mask_25_evidence = get_input_masks_for_claim_and_evidence(b_input_mask, b_input_ids)            

                outputs = extractor_model(cls_last_hidden_state).view(b_input_ids.shape[0], -1, 2)
                outputs = F.gumbel_softmax(outputs, tau=0.4, hard=True, dim=-1)[:, :, -1]
                # outputs = F.softmax(outputs, dim=-1)[:, :, -1]
                
                outputs_inv = torch.ones_like(outputs) - outputs
                
                b_input_mask_use_extractor = torch.bmm(outputs.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim
                b_input_mask_use_extractor_inv = torch.bmm(outputs_inv.unsqueeze(1), b_input_mask_25_evidence.float()).squeeze(1) + b_input_mask_claim

                outputs_use_extractor = bert_finetuned(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask_use_extractor, 
                    labels=b_labels
                )
                loss_use_extractor, logits = outputs_use_extractor[0], outputs_use_extractor[1]
                # torch.cuda.empty_cache()

                outputs_use_extractor_inv = bert_finetuned(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask_use_extractor_inv, 
                    labels=b_labels
                )
                loss_use_extractor_inv = outputs_use_extractor_inv[0]
                # torch.cuda.empty_cache()
                
                loss = loss_use_extractor - loss_use_extractor_inv + loss_plau
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            all_domains = np.concatenate((all_domains, b_domains), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print('Validation Elapsed: {:}.'.format(validation_time))
        # print(bert_type)
        c = Counter()
        for pred in all_prediction:
            c[int(pred)] += 1
        print(c)
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.2%}".format(f1))
        microf1 = f1
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.2%}".format(pre))
        print("   Recall (macro): {:.2%}".format(recall))
        print("       F1 (macro): {:.2%}".format(f1))
        if f1 > best_macrof1:
            best_microf1 = microf1
            best_macrof1 = f1
            best_recall = recall
            best_precision = pre
            print('Above is best')
            # display every label's f1 score
            pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction)
            print("Precision:", pre)
            print("   Recall:", recall)
            print("       F1:", f1)
            # calculate every domain's f1 score
            all_domains = np.around(all_domains).astype(int)
            # there are 5 domains
            domains_num = 5
            ground_truths = [[] for i in range(domains_num)]
            predictions = [[] for i in range(domains_num)]
            """
            for i in range(len(all_ground_truth)):
                ground_truths[all_domains[i]].append(all_ground_truth[i])
                predictions[all_domains[i]].append(all_prediction[i])
            for i in range(domains_num):
                print('length id', i, 'score: ')
                pre, recall, f1, _ = precision_recall_fscore_support(ground_truths[i], predictions[i], average='micro')
                print("       F1 (micro): {:.2%}".format(f1))
                pre, recall, f1, _ = precision_recall_fscore_support(ground_truths[i], predictions[i], average='macro')
                print("Precision (macro): {:.2%}".format(pre))
                print("   Recall (macro): {:.2%}".format(recall))
                print("       F1 (macro): {:.2%}".format(f1))
                print('================' * 3)
            """
        print("****************************")
        print("Best Validation Record:")
        print("       F1 (micro): {:.2%}".format(best_microf1))
        print("Precision (macro): {:.2%}".format(best_precision))
        print("   Recall (macro): {:.2%}".format(best_recall))
        print("       F1 (macro): {:.2%}".format(best_macrof1))
        print("****************************")
        torch.save(extractor_model.state_dict(), './evidenceextractor.bin')
        torch.save(bert_model.state_dict(), './bertforseqcls_for_ext.bin')

    # save model as "bertforseqcls.bin"

    pdb.set_trace()
    return best_microf1, best_precision, best_recall, best_macrof1

def redo_attention_mask(b_input_ids, b_attention_mask, mode=1):
    pdb.set_trace()

def get_input_masks_for_claim_and_evidence(b_input_mask, b_input_ids):
    # will use evidence extractor, and alter b_input_mask
    b_input_mask_claim = torch.zeros_like(b_input_mask)
    for bz in range(b_input_mask.shape[0]):
        for idx in range(b_input_mask.shape[1]):
            b_input_mask_claim[bz][idx] = 1
            if b_input_ids[bz][idx] == 102:
                break
        
    # make tensor or shape(b_input_mask[0], 25, b_input_mask[1]), init with all zero.
    b_input_mask_25_evidence = torch.zeros([b_input_mask.shape[0], 25, b_input_mask.shape[1]], dtype=torch.long).to(device)
    for bz in range(b_input_mask.shape[0]):
        count = -1
        for idx in range(b_input_mask.shape[1]):
            if count >= 25:
                # this token, and any other token after this, is not evidence
                # so leave it as zero
                break
            if count >= 0:
                b_input_mask_25_evidence[bz][count][idx] = 1
            if b_input_ids[bz][idx] == 102:
                count += 1
    return b_input_mask_claim, b_input_mask_25_evidence

if __name__ == '__main__':
    sys.exit(main())
    