#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/agentbelongstoorganization/"
vocab_dir="datasets/data_preprocessed/agentbelongstoorganization/vocab"
total_iterations=500
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.07
Lambda=0.07
use_entity_embeddings=1
use_cluster_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/agentbelongstoorganization/"
load_model=0
model_load_dir="saved_models/agentbelongstoorganization"
nell_evaluation=1
learning_rate=1e-3
gamma=0.5