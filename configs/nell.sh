#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell/"
vocab_dir="datasets/data_preprocessed/nell/vocab"
total_iterations=3000
path_length=5
hidden_size=50
embedding_size=50
batch_size=128
beta=0.07
Lambda=0.07
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/nell/"
load_model=0
model_load_dir="saved_models/nell"
nell_evaluation=0
