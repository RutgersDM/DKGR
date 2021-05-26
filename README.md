# Curl
Learning to Walk with Dual Agents for Knowledge Graph Reasoning


## Requirements
To install the various python dependencies (including pytorch)
```
pip install -r requirements.txt
```

## Training & Testing
The hyperparam configs for each experiments are included in the [configs](https://github.com/yuanzx33033/Curl/tree/master/configs) directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the `${dataset}.sh` is the name of the config file. For example, 
```
sh run.sh configs/nell.sh
```

## Output
The code outputs the evaluation of Curl on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20}, MRR, and MAP.  Along with this, the code also outputs the answers Curl reached in a file.
