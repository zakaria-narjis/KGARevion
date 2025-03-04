import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import math
import argparse
import torch
from transformers import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch_geometric.data import Data
from torch_geometric.data import Batch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import pandas as pd
import numpy as np
import random
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel

class LoadPrimeKG(object):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.kg_path = data_path + "kg.csv"
        self.entities_path = data_path + "entities.txt"
        self.num_node, self.num_rel, self.samples, self.edge_index = self.read_kg()

    def read_entity_dict(self, entity_index, entity_name, rel):
        entity_dict = {}
        entity_type = {}
        for (e_index, e_n, e_rel) in zip(entity_index, entity_name, rel):
            if e_index not in entity_dict:
                entity_dict[e_index] = e_n  ##index in KG and id in dataset
            if e_rel not in entity_type:
                entity_type[e_rel] = [e_index]
            else:
                entity_type[e_rel].append(e_index)

        print("PrimeKG has {} entities.".format(len(entity_dict)))

        return entity_dict, entity_type

    def read_rel_dict(self, rel):
        rel_dict = {}
        for r in rel:
            if r not in rel_dict:
                rel_dict[r] = len(rel_dict)
        return rel_dict

    def construct_kg_dict(self, head_entity_index, tail_entity_index):
        kg_dict = {}
        for (h,t) in zip(head_entity_index, tail_entity_index):
            if h not in kg_dict:
                kg_dict[h] = [t]
            else:
                kg_dict[h].append(t)
        return kg_dict

    def negative_samples(self, h, r):
        for e in self.entity_type_dict:
            if r == e:
                rand_idx = random.randint(0, len(self.entity_type_dict[e])-1)
                print("rand_idx is {}".format(rand_idx))
                while self.entity_type_dict[e][rand_idx] in self.kg_dict[h]:
                    rand_idx = random.randint(0, len(self.entity_type_dict[e])-1)
                return self.entity_type_dict[e][rand_idx]

    def read_kg(self):
        kg = pd.read_csv(self.kg_path, sep=',', dtype=str)

        #relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source
        head_entity_index, head_entity_id, head_entity_type, head_entity_name = kg['x_index'].tolist(), kg['x_id'].tolist(), kg['x_type'].tolist(), kg['x_name'].tolist()
        tail_entity_index, tail_entity_id, tail_entity_type, tail_entity_name = kg['y_index'].tolist(), kg['y_id'].tolist(), kg['y_type'].tolist(), kg['y_name'].tolist()
        rel = kg['relation'].tolist()

        head_entity_index = [int(i) for i in head_entity_index]
        tail_entity_index = [int(i) for i in tail_entity_index]
        print(len(head_entity_index))

        entity_index = head_entity_index + tail_entity_index
        entity_name = head_entity_name + tail_entity_name
        entity_rel = rel + rel
        print(len(entity_index))

        num_node = len(set(entity_index))

        assert len(entity_index) == len(entity_name)
        assert len(entity_name) == len(entity_rel)

        self.entity_name_dict, self.entity_type_dict = self.read_entity_dict(entity_index, entity_name, entity_rel)
        self.kg_dict = self.construct_kg_dict(head_entity_index, tail_entity_index)

        ##create the negative samples
        if os.path.exists(self.data_path + "samples.txt"):
            samples = []
            edge_index = []
            rel_index = {}
            with open(self.data_path + "samples.txt") as f:
                for line in f.readlines():
                    h, t, t_neg, r = line.strip().split("\t")
                    if r not in rel_index:
                        rel_index[r] = len(rel_index)
                    samples.append([int(h), int(t), int(t_neg), int(rel_index[r])])
                    edge_index.append([int(h), int(t)])
                    
            num_rel = len(rel_index)
        else:
            samples = []
            for (h,t,r) in zip(head_entity_index, tail_entity_index, rel):
                #print("positive samples are {} {} {}".format(h,t,r))
                t_neg = self.negative_samples(h, r)
                #print("negative samples are {} {} {}".format(h,t_neg, r))
                samples.append([h, t, t_neg, r])
                #print(len(samples))

            with open(self.data_path + "samples.txt", 'w') as f:
                for s in samples:
                    f.write(str(s[0]))
                    f.write("\t")
                    f.write(str(s[1]))
                    f.write("\t")
                    f.write(str(s[2]))
                    f.write("\t")
                    f.write(str(s[3]))
                    f.write("\n")
                f.close()

        return num_node, num_rel+1, samples, edge_index

class PrimeKGDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.sample = data.samples
        self.entity_name = data.entity_name_dict

    def tokenize(self, rel):
        
        entity_data = Data(
            x = torch.LongTensor([int(rel)])
        )
        return entity_data

    def __getitem__(self, index):
        sample = self.sample[index]
        rel = sample[-1]
        h_data = self.tokenize(sample[0])
        t_data = self.tokenize(sample[1])
        t_n_data = self.tokenize(sample[2])
        r_data = self.tokenize(rel)

        return h_data, t_data, t_n_data, r_data


    def __len__(self):
        return len(self.sample)

def my_collate(batch):
    head_entity = Batch.from_data_list([data[0] for data in batch])
    tail_entity = Batch.from_data_list([data[1] for data in batch])
    tail_neg_entity = Batch.from_data_list([data[2] for data in batch])
    rel = Batch.from_data_list([data[3] for data in batch])

    return head_entity, tail_entity, tail_neg_entity, rel

class PreEmbed(KGEModel):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.margin = margin
        self.node_emb_im = Embedding(num_nodes, hidden_channels, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
        torch.nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        head_index = head_index.x
        rel_type = rel_type.x
        tail_index = tail_index.x

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        rel_theta = self.rel_emb(rel_type)
        #print(rel_theta)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin-score

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        tail_neg_index: Tensor
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(head_index, rel_type, tail_neg_index)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)


set_seed(42)
data = LoadPrimeKG(data_path="primeKG/")

num_entities = data.num_node
num_relations = data.num_rel

# Parameters
embedding_dim = 128  # Embedding dimension
num_epochs = 100
batch_size = 1024
learning_rate = 0.005

primeKG_dataset = PrimeKGDataset(data = data)
primeKG_dataloader = DataLoader(primeKG_dataset, batch_size = 512, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, collate_fn=my_collate)

# Initialize model, loss function, and optimizer
model = PreEmbed(num_entities, num_relations, embedding_dim)
model.to('cuda')
loss_function = nn.MarginRankingLoss(margin=1.0).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
min_loss = 10000
stop_sign = 0
for epoch in range(num_epochs):
    total_loss = 0
    epoch_iterator = tqdm(primeKG_dataloader, desc=f'Epoch[{epoch}/{num_epochs}]', ascii=True)
    for _, data in enumerate(epoch_iterator):
        head_entity = data[0].to('cuda')
        tail_entity = data[1].to('cuda')
        tail_neg_entity = data[2].to('cuda')
        rel = data[3].to('cuda')
            
        optimizer.zero_grad()

        loss = model.loss(head_entity, rel, tail_entity, tail_neg_entity)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(epoch_iterator)}')
    
    if epoch > 4 and (total_loss/len(epoch_iterator)) < min_loss:
        min_loss = total_loss/len(epoch_iterator)
        stop_sign = 0
        torch.save(model.state_dict(), 'prime_rotate_new.pth')
        print("save model!")
    elif epoch > 4 and (total_loss/len(epoch_iterator))  > min_loss:
        stop_sign += 1
        if stop_sign > 5:
            break

