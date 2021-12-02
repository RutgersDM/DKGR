from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityAndClusterBatcher
from code.data.grapher import RelationEntityGrapher, RelationClusterGrapher
import logging
import torch

logger = logging.getLogger()


class EntityEpisode(object):

    def __init__(self, graph, data, params, mode):
        self.grapher = graph

        self.pretrained_entity_embeddings = params['pretrained_embeddings_entity']
        self.pretrained_relation_embeddings = params['pretrained_embeddings_relation']

        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = params['num_rollouts']
        else:
            self.num_rollouts = params['test_rollouts']

        self.batch_size = params['batch_size']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']

        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']

        self.entity_id_to_cluster_mappping = params['entity_id_to_cluster_mappping']
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers = data
        self.no_examples = start_entities.shape[0] # original batch_size

        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts) # [ent1, ent1, ..., ent2, ent2, ..., ent3, ent3, ...]

        self.cluster_path = {}
        self.approximated_reward = {}
        self.e_agent_cls_path = {}
        for i, ent in enumerate(start_entities):
            self.cluster_path[i] = [self.entity_id_to_cluster_mappping[str(ent)]]
            self.approximated_reward[i] = []
        for p_len in range(self.path_len):
            self.e_agent_cls_path[p_len] = []
        self.credits = []

        # print('length: ', len(self.e_agent_cls_path[0]))

        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = start_entities
        self.query_relation = batch_query_relation
        self.all_answers = all_answers


        next_actions = self.grapher.init_actions(self.current_entities, self.start_entities, self.query_relation,
                                                            self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                            self.num_rollouts)

        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1] # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_entities'] = next_actions[:, :, 0]

        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [original batch_size * num_rollout]

        # return reward
        return reward

    def get_stepwise_approximated_reward(self, current_entities, current_clusters, prev_entities):

        credit = []
        num_rollout = int(current_entities.size(0) / self.batch_size)
        current_entities = current_entities.cpu().numpy()
        # prev_entities = prev_entities.cpu().numpy()
        for i in range(0, len(current_entities), num_rollout):
            correct_num = 0.0
            num = 0.0
            for j in range(num_rollout):
                idx = i+j
                curr_ent = current_entities[idx]
                try:
                    ent2cls = self.entity_id_to_cluster_mappping[str(curr_ent)]
                except:
                    continue
                curr_cls = current_clusters[idx]
                if curr_ent != 0:
                    num += 1.0
                    if ent2cls == curr_cls:
                        correct_num += 1.0
            if num == 0.0:
                credit.append(0.0)
            else:
                credit.append(correct_num/num)

        credit = torch.repeat_interleave(torch.tensor(credit), num_rollout)
        self.credits.append(credit)



    def __call__(self, action, prev_cluster, p_len):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action.cpu()]

        for i, cls in enumerate(prev_cluster): self.cluster_path[i].append(cls)

        next_actions, self.cluster_path, whether_e_agent_follows_c_agent, self.e_agent_cls_path = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                                                                  self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                                                                  self.num_rollouts, self.cluster_path, self.e_agent_cls_path, p_len)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

        return self.state, whether_e_agent_follows_c_agent


class ClusterEpisode(object):

    def __init__(self, graph, data, params, mode):
        self.grapher = graph

        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = params['num_rollouts']
        else:
            self.num_rollouts = params['test_rollouts']

        self.batch_size = params['batch_size']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']

        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']

        self.current_hop = 0

        start_clusters, end_entities, all_answers = data
        self.no_examples = start_clusters.shape[0]  # original batch_size

        start_clusters = np.repeat(start_clusters, self.num_rollouts)
        end_clusters = np.repeat(end_entities, self.num_rollouts)  # [cls1, cls1, ..., cls2, cls2, ..., cls3, cls3, ...]

        self.start_clusters = start_clusters
        self.end_clusters = end_clusters
        self.current_clusters = start_clusters
        self.all_answers = all_answers

        next_actions = self.grapher.return_next_actions_cluster(self.current_clusters, self.start_clusters,
                                                                self.end_clusters, self.all_answers,
                                                                self.current_hop == self.path_len - 1,
                                                                self.num_rollouts)
        self.state = {}
        self.state['next_cluster_relations'] = next_actions[:, :, 1]  # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_clusters'] = next_actions[:, :, 0]

        self.state['current_clusters'] = self.current_clusters

    def get_reward(self):
        reward = (self.current_clusters == self.end_clusters)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [original batch_size * num_rollout]

        return reward

    def get_query_cluster_relation(self):
        return self.end_clusters

    def get_state(self):
        return self.state

    def next_action(self, action):
        self.current_hop += 1
        self.current_clusters = self.state['next_clusters'][np.arange(self.no_examples * self.num_rollouts), action.cpu()]

        next_actions = self.grapher.return_next_actions_cluster(self.current_clusters, self.start_clusters,
                                                                self.end_clusters, self.all_answers,
                                                                self.current_hop == self.path_len - 1, self.num_rollouts)

        self.state['next_cluster_relations'] = next_actions[:, :, 1]  # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_clusters'] = next_actions[:, :, 0]
        self.state['current_clusters'] = self.current_clusters
        return self.state



class env(object):
    def __init__(self, params, mode='train'):

        self.params = params
        self.mode = mode

        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityAndClusterBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 cluster_vocab=params['cluster_vocab'],
                                                 cluster_relation_vocab=params['cluster_relation_vocab'],
                                                 entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping']
                                                 )
        else:
            self.batcher = RelationEntityAndClusterBatcher(input_dir=input_dir,
                                                           mode=mode,
                                                           batch_size=params['batch_size'],
                                                           entity_vocab=params['entity_vocab'],
                                                           relation_vocab=params['relation_vocab'],
                                                           cluster_vocab=params['cluster_vocab'],
                                                           cluster_relation_vocab=params['cluster_relation_vocab'],
                                                           entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping']
                                                           )

            self.total_no_examples = self.batcher.store.shape[0]

        self.entity_grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                                    max_num_actions=params['max_num_actions'],
                                                    entity_vocab=params['entity_vocab'],
                                                    relation_vocab=params['relation_vocab'],
                                                    entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping'],
                                                    )

        self.cluster_grapher = RelationClusterGrapher(cluster_vocab=params['cluster_vocab'],
                                                      cluster_relation_vocab=params['cluster_relation_vocab']
                                                      )

    def get_episodes(self, batch_counter):

        if self.mode == 'train':
            for entity_data, cluster_data in self.batcher.yield_next_batch_train():
                yield EntityEpisode(self.entity_grapher, entity_data, self.params, self.mode), ClusterEpisode(self.cluster_grapher, cluster_data, self.params, self.mode)
        else:
            for entity_data, cluster_data in self.batcher.yield_next_batch_test():
                if entity_data == None or cluster_data == None:
                    return
                yield EntityEpisode(self.entity_grapher, entity_data, self.params, self.mode), ClusterEpisode(self.cluster_grapher, cluster_data, self.params, self.mode)
