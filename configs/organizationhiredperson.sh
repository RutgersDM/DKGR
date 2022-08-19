#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/organizationhiredperson/"
vocab_dir="datasets/data_preprocessed/organizationhiredperson/vocab"
total_iterations=100
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.07
Lambda=0.07
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/organizationhiredperson/"
load_model=0
model_load_dir="saved_models/organizationhiredperson"
nell_evaluation=1
learning_rate=1e-3
gamma=0.5