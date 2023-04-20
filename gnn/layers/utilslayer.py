# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:38:31 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

from keras.layers import Layer
from keras import backend as K

class ReceFieldEncoder(Layer):
    def __init__(self, config, name = 'receptive_field_encoder', **kwarg):
        super(ReceFieldEncoder, self).__init__(name = name, **kwarg)
        self.config = config
    
    def call(self, entity):
        """Calculate receptive field for entity using adjacent matrix
        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list


class NeighborEmbedder(Layer):
    def __init__(self, config, name = 'neighbor_embedder', **kwarg):
        super(NeighborEmbedder, self).__init__(name = name, **kwarg)
        self.config = config
    
    def call(self, embeddings):
        """Get neighbor representation.
        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        drug, rel, ent = embeddings
        
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = K.sum(drug * rel, axis=-1, keepdims=True)

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed
    

class SqueezeLayer(Layer):
    def __init__(self, name = 'squeeze_layer', **kwarg):
        super(SqueezeLayer, self).__init__(name = name, **kwarg)
    
    def call(self, embedding):
        return K.squeeze(embedding, axis=1)
    

class ScorerLayer(Layer):
    def __init__(self, name = 'scorer_layer', **kwarg):
        super(ScorerLayer, self).__init__(name = name, **kwarg)
    
    def call(self, embeddings, scorer = 'sigmoid'):
        embedding1, embedding2 = embeddings[0], embeddings[1]
        if scorer == 'sigmoid':
            return K.sigmoid(K.sum(embedding1 * embedding2, axis=-1, keepdims=True))

    
        