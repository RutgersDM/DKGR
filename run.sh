#!/bin/bash

. $1
export PYTHONPATH="."
gpu_id=0
cmd="python3 trainer.py --learning_rate $learning_rate --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
    --batch_size $batch_size --beta $beta --Lambda $Lambda --use_entity_embeddings $use_entity_embeddings --use_cluster_embeddings $use_cluster_embeddings\
    --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
    --data_input_dir $data_input_dir --vocab_dir $vocab_dir --model_load_dir $model_load_dir --load_model $load_model --total_iterations $total_iterations\
    --nell_evaluation $nell_evaluation --learning_rate $learning_rate --gamma $gamma"



echo "Executing $cmd"

CUDA_VISIBLE_DEVICES=$gpu_id $cmd
