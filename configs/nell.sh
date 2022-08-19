#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell/"
vocab_dir="datasets/data_preprocessed/nell/vocab"
total_iterations=10000
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.12
Lambda=0.12
use_entity_embeddings=1
use_cluster_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/nell/"
load_model=0
model_load_dir="saved_models/nell"
nell_evaluation=0
learning_rate=1e-2
gamma=0.2