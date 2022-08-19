import json
import csv
import argparse
import os
from collections import defaultdict


def entity_level_vocab():
    root_dir = '../../../'
    vocab_dir = root_dir+'datasets/data_preprocessed/FB15K-237/vocab/'
    dir = root_dir+'datasets/data_preprocessed/FB15K-237/'

    entity_vocab = {}
    relation_vocab = {}

    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)

    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)

    if os.path.isfile(dir + 'full_graph.txt'):
        fact_files = ['full_graph.txt']
        print("Contains full graph")
    else:
        fact_files = ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']

    for f in fact_files:
        with open(dir+f) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:

                e1, r, e2 = line

                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1

    with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
        json.dump(entity_vocab, fout)

    with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
        json.dump(relation_vocab, fout)

# entity_level_vocab()

def cluster_level_vocab():
    root_dir = '../../../'
    vocab_dir = root_dir + 'datasets/data_preprocessed/FB15K-237/vocab/' # FB15K-237
    dir = root_dir + 'datasets/data_preprocessed/FB15K-237/'

    entity_vocab = json.load(open(vocab_dir + 'entity_vocab.json', 'r'))
    # os.makedirs(vocab_dir)

    f1 = open(os.path.join(dir, 'entity2clusterid.txt'))
    ent2cluster = f1.readlines()

    f2 = open(os.path.join(dir, 'entity2id.txt'))
    entity2id = f2.readlines()

    ent2cluster_ = {}
    entity2id_ = {}

    for line in entity2id:
        entity2id_[line.split()[0]] = int(line.split()[1])

    for line in ent2cluster:
        ent2cluster_[int(line.split()[0])] = str(line.split()[1])

    num_cls = len(set(list(ent2cluster_.values())))
    print(num_cls)

    cluster_vocab = {}
    relation_vocab = {}

    cluster_vocab['PAD'] = len(cluster_vocab)
    cluster_vocab['UNK'] = len(cluster_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    # relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)

    cluster_counter = len(cluster_vocab)
    relation_counter = len(relation_vocab)
    entity_id_to_cluster_mappping = {}

    for c in range(num_cls):
        cluster_vocab[str(c)] = cluster_counter
        cluster_counter += 1

    # print(len(cluster_vocab))

    if os.path.isfile(dir + 'full_graph.txt'):
        fact_files = ['full_graph.txt']
        print("Contains full graph")
    else:
        fact_files = ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']

    for f in fact_files:
        with open(dir + f) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:
                e1, r, e2 = line
                c1 = cluster_vocab[ent2cluster_[entity2id_[e1]]]
                c2 = cluster_vocab[ent2cluster_[entity2id_[e2]]]

                c_rel = str(c1) + '_' + str(c2)

                e1_id = entity_vocab[e1]
                e2_id = entity_vocab[e2]

                if e1_id not in entity_id_to_cluster_mappping:
                    entity_id_to_cluster_mappping[e1_id] = c1
                else:
                    assert entity_id_to_cluster_mappping[e1_id] == c1

                if e2_id not in entity_id_to_cluster_mappping:
                    entity_id_to_cluster_mappping[e2_id] = c2
                else:
                    assert entity_id_to_cluster_mappping[e2_id] == c2

                if c_rel not in relation_vocab and c_rel is not None:
                    relation_vocab[c_rel] = relation_counter
                    relation_counter += 1

    for i in range(num_cls+2):
        c_rel = str(i) + '_' + str(i)
        if c_rel not in relation_vocab:
            relation_vocab[c_rel] = relation_counter
            relation_counter += 1

    with open(vocab_dir + 'cluster_vocab.json', 'w') as fout:
        json.dump(cluster_vocab, fout)

    with open(vocab_dir + 'cluster_relation_vocab.json', 'w') as fout:
        json.dump(relation_vocab, fout)

    with open(vocab_dir + 'entity_id_to_cluster_mappping.json', 'w') as fout:
        json.dump(entity_id_to_cluster_mappping, fout)

    print(vocab_dir + 'entity_id_to_cluster_mappping.json')

cluster_level_vocab()