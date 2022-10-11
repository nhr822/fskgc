import logging

import torch
import torch.nn as nn

from modules import *


class Sub_encoder(nn.Module):
    def __init__(self, embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain, ent_embed=None,
                 dropout_input=0.3, finetune=False, dropout_neighbors=0.3):
        super(Sub_encoder, self).__init__()
        self.structure_encoder = Att_GCN(embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain,
                                         ent_embed,  dropout_input, finetune, dropout_neighbors)

    def forward(self, hrt, neb, nebr, adj):         # 5,3   5,19    5,19,5      5,20,20
        h_em, t_em, r_em, tri_em = self.structure_encoder(hrt, neb, nebr, adj)
        return h_em, t_em, r_em, tri_em


class Enhance_Rel(nn.Module):
    def __init__(self, emb_dim, device, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(Enhance_Rel, self).__init__()
        self.RelationEncoder = Mapping(device, model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                       num_heads=num_transformer_heads, dropout=dropout_rate,
                                       num_layers=num_transformer_layers, max_seq_len=3, with_pos=True)

    def forward(self, h_em, t_em, r_em, triple_ht):
        relation = self.RelationEncoder(h_em, t_em, r_em, triple_ht)
        return relation


class fskgc(nn.Module):
    def __init__(self, embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain=True, ent_embed=None,
                 dropout_layers=0.1, dropout_input=0.3, dropout_neighbors=0.0, finetune=False, num_transformer_layers=6,
                 num_transformer_heads=4):
        super(fskgc, self).__init__()
        self.Encoder = Sub_encoder(embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain=use_pretrain,
                                   ent_embed=ent_embed, dropout_input=dropout_input,
                                   dropout_neighbors=dropout_neighbors, finetune=finetune)
        self.enhance_Rel = Enhance_Rel(embed_dim, device, num_transformer_layers=num_transformer_layers,
                                        num_transformer_heads=num_transformer_heads, dropout_rate=dropout_layers)
        self.Prototype = ACPrototype(embed_dim * 2)

    def forward(self, hrt_list, neb_list, nebr_list, adj_list, isEval=False):
        if not isEval:
            s_h_em, s_t_em, s_r_em, support_tri = self.Encoder(hrt_list[0], neb_list[0], nebr_list[0], adj_list[0])
            q_h_em, q_t_em, q_r_em, query_tri = self.Encoder(hrt_list[1], neb_list[1], nebr_list[1], adj_list[1])
            f_h_em, f_t_em, f_r_em, false_tri = self.Encoder(hrt_list[2], neb_list[2], nebr_list[2], adj_list[2])

            support_r = self.enhance_Rel(s_h_em, s_t_em, s_r_em, support_tri)        # 5, 100
            query_r = self.enhance_Rel(q_h_em, q_t_em,  q_r_em, query_tri)
            false_r = self.enhance_Rel(f_h_em, f_t_em,  f_r_em, false_tri)

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)

            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:
            s_h_em, s_t_em, s_r_em, support_tri = self.Encoder(hrt_list[0], neb_list[0],nebr_list[0], adj_list[0])
            q_h_em, q_t_em, q_r_em, query_tri = self.Encoder(hrt_list[1], neb_list[1],nebr_list[1], adj_list[1])

            support_r = self.enhance_Rel(s_h_em, s_t_em, s_r_em,support_tri)        # 5, 400
            query_r = self.enhance_Rel(q_h_em, q_t_em, q_r_em, query_tri)

            center_q = self.Prototype(support_r, query_r)

            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
        return positive_score, negative_score
