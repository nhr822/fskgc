from collections import defaultdict
from torch import optim
from collections import deque
from args import read_options
from data_loader import *
from model import *
# from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp


class Trainer(object):
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)
        self.data_file = os.path.abspath('..') + "/data/" + self.dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # pre-train
        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        # if self.test or self.random_embed:
        #     # gen symbol2id, without embedding
        use_pretrain = False
        # else:
        #     self.load_embed()
        self.use_pretrain = use_pretrain

        self.ent2id = json.load(open(self.data_file + '/ent2ids'))
        self.rel2id = json.load(open(self.data_file + '/rel2id'))         # relation2ids
        self.num_ents = len(self.ent2id.keys())
        self.num_ents_pad = len(self.ent2id.keys())+self.max_neighbor
        # self.num_ent_pad = self.max_neighbor
        self.num_rels = len(self.rel2id.keys())
        self.num_rels_pad = len(self.rel2id.keys())+self.max_neb_rel
        self.batch_nums = 0

        self.fskgc = fskgc(self.embed_dim, self.num_ents_pad, self.num_rels_pad, self.max_neighbor, self.device,
                               use_pretrain=self.use_pretrain, finetune=self.fine_tune)
        self.fskgc.to(self.device)

        self.parameters = filter(lambda p: p.requires_grad, self.fskgc.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.data_file + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.data_file + '/e1rel_e2.json'))

    # def load_embed(self):
    #     # gen symbol2id, with embedding
    #     symbol_id = {}
    #     rel2id = json.load(open(self.data_file + '/relation2ids'))  # relation2id contains inverse rel    rel2id
    #     ent2id = json.load(open(self.data_file + '/ent2ids'))
    #
    #     logging.info('LOADING PRE-TRAINED EMBEDDING')
    #     if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
    #         ent_embed = np.loadtxt(self.data_file + '/entity2vec.' + self.embed_model)
    #         # rel_embed = np.loadtxt(self.data_file + '/relation2vecs.' + self.embed_model)
    #         if self.embed_model == 'ComplEx':
    #             # normalize the complex embeddings
    #             ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
    #             ent_std = np.std(ent_embed, axis=1, keepdims=True)
    #             eps = 1e-3
    #             ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
    #
    #             # rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
    #             # rel_std = np.std(rel_embed, axis=1, keepdims=True)
    #             # eps = 1e-3
    #             # rel_embed = (rel_embed - rel_mean) / (rel_std + eps)
    #
    #         # assert ent_embed.shape[0] == len(ent2id.keys())
    #         # assert rel_embed.shape[0] == len(rel2id.keys())
    #
    #         ent_embeddings = []
    #         for key in ent2id.keys():
    #             if key not in ['', 'OOV']:
    #                 ent_embeddings.append(list(ent_embed[ent2id[key], :]))
    #         ent_embeddings = np.array(ent_embeddings)
    #         self.ent2vec = ent_embeddings
    #
    #         # rel_embeddings = []
    #         # for key in rel2id.keys():
    #         #     if key not in ['', 'OOV']:
    #         #         rel_embeddings.append(list(rel_embed[rel2id[key], :]))
    #         # rel_embeddings = np.array(rel_embeddings)
    #         # self.rel2vec = rel_embeddings

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.fskgc.state_dict(), path)

    def load(self, path=None):
        if path:
            self.fskgc.load_state_dict(torch.load(path))
        else:
            self.fskgc.load_state_dict(torch.load(self.save_path))

    def graph_build(self):
        self.e1_neb = defaultdict(list)       # 实体： 邻居1, 邻居2...
        self.r1_neb = defaultdict(list)       # 实体： 邻居1, 邻居2...
        with open(self.data_file + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_neb[self.ent2id[e1]].append(self.ent2id[e2])  # {头实体id:[尾实体id]}
                self.e1_neb[self.ent2id[e2]].append(self.ent2id[e1])  # {头实体id:[尾实体id]}
                self.r1_neb[self.ent2id[e1]].append(self.rel2id[rel])  # {头实体id:[尾实体id]}
                self.r1_neb[self.ent2id[e2]].append(self.rel2id[rel])  # {头实体id:[尾实体id]}

    def get_h_BFS(self, G, RG, beg_node):
        # beg_node = 41728        # 41728
        # beg_node = [29592, 45100]
        beg_node_h = beg_node[0]
        beg_node_t = beg_node[2]
        beg_node_str = beg_node_h+beg_node_t+10000000
        # subgraph0_list = [beg_node_h, beg_node_t]       # [hid,tid]
        subgraph0_list = []       # [hid,tid]

        e_neb_dict = defaultdict(list)
        nei_ht_1 = G[beg_node_h] + G[beg_node_t]
        subgraph0_list.extend(nei_ht_1)                   # [hid, tid, nei1, nei2]
        subgraph_list = []
        for i in subgraph0_list:                            # 去重
            if i not in subgraph_list:
                subgraph_list.append(i)                     # [hid, tid, nei1, nei2]
        # print("subgraph_list1", subgraph_list)
        if len(subgraph_list) >= self.max_neighbor:               # 一阶邻居大于最大邻居个数，就从一阶邻居里选
            subgraph_list = subgraph_list[:self.max_neighbor]
            for i in subgraph_list:
                e_neb_dict[beg_node_str].append(i)
        else:
            neb1_num = len(subgraph_list)                # 此时邻居个数 =  起始节点  加  一阶邻居的个数
            for i in subgraph_list:
                e_neb_dict[beg_node_str].append(i)             # 首先把 起始节点：[一阶邻居] 加入字典
            # print("e_neb_dict1", e_neb_dict)
            all_neb2_list = []
            for n1 in subgraph_list:                              # 一阶邻居里的每个邻居  [2:]
                nei2_li = list(set(i for i in G[n1] if i not in subgraph_list))  # 第n个一阶邻居节点的二阶邻居
                if len(nei2_li) != 0:
                    all_neb2_list.extend(nei2_li)                   # 二阶邻居
                    neb_num = len(nei2_li) + neb1_num               # 总个数 = 第n个节点的邻居个数 加 此时列表邻居个数
                    if neb_num >= self.max_neighbor:                     # 总个数 大于 最大邻居个数
                        for i in nei2_li[:self.max_neighbor-neb1_num]:
                            e_neb_dict[n1].append(i)                # 对应字典
                        subgraph_list.extend(e_neb_dict[n1])       # 从二阶邻居 里 选出 最大个数 - 一阶林觉个数
                        break
                    else:
                        for i in nei2_li:
                            e_neb_dict[n1].append(i)
                        subgraph_list.extend(e_neb_dict[n1])       # 从二阶邻居 里 选出 最大个数-一阶林觉个数
                        neb1_num = len(subgraph_list)              # 此时邻居个数设置为 第n个节点的邻居个数 加 一阶邻居个数
                else:
                    pass
            # print("all_neb2_list", all_neb2_list)
            # print("subgraph_list2", subgraph_list)
            # print("e_neb_dict2", e_neb_dict)
            # 循环完了 一阶邻居，找到了所有二阶邻居， 此时列表邻居个数 < 最大邻居个数
            # print(subgraph_list)
            neb2_num = len(subgraph_list)
            all_neb3_list = []
            if neb2_num < self.max_neighbor:
                for n2 in all_neb2_list:                              # 一阶邻居里的每个邻居
                    nei3_li = list(set(i for i in G[n2] if i not in subgraph_list))  # 第n个二阶邻居节点的三阶邻居
                    if len(nei3_li) !=0:
                        all_neb3_list.extend(nei3_li)
                        neb_num = len(nei3_li) + neb2_num               # 总个数 = 第n个节点的邻居个数 加 此时列表邻居个数
                        if neb_num >= self.max_neighbor:                     # 总个数 大于 最大邻居个数
                            for i in nei3_li[:self.max_neighbor-neb2_num]:
                                e_neb_dict[n2].append(i)            # 对应字典
                            subgraph_list.extend(e_neb_dict[n2])       # 从二阶邻居 里 选出 最大个数 - 此时列表邻居个数
                            break
                        else:
                            for i in nei3_li:
                                e_neb_dict[n2].append(i)
                            subgraph_list.extend(e_neb_dict[n2])       # 从二阶邻居 里 选出 最大个数-一阶林觉个数
                            neb2_num = len(subgraph_list)              # 此时邻居个数设置为 第n个节点的邻居个数 加 此时邻居个数
                    else:
                        pass
                # print("all_neb3_list", all_neb2_list)
                # print("subgraph_list3", subgraph_list)
                # print("e_neb_dict3", e_neb_dict)

                # 循环完了 一阶邻居，找到了所有二阶邻居， 此时列表邻居个数 < 最大邻居个数
                neb3_num = len(subgraph_list)
                all_neb4_list = []
                if neb3_num < self.max_neighbor:
                    for n3 in all_neb3_list:  # 一阶邻居里的每个邻居
                        nei4_li = list(set(i for i in G[n3] if i not in subgraph_list))  # 第n个二阶邻居节点的三阶邻居
                        all_neb4_list.extend(nei4_li)
                        if len(nei4_li) != 0:
                            neb_num = len(nei4_li) + neb3_num  # 总个数 = 第n个节点的邻居个数 加 此时列表邻居个数
                            if neb_num >= self.max_neighbor:  # 总个数 大于 最大邻居个数
                                for i in nei4_li[:self.max_neighbor - neb3_num]:
                                    e_neb_dict[n3].append(i)  # 对应字典
                                subgraph_list.extend(e_neb_dict[n3])  # 从二阶邻居 里 选出 最大个数 - 此时列表邻居个数
                                break
                            else:
                                for i in nei4_li:
                                    e_neb_dict[n3].append(i)
                                subgraph_list.extend(e_neb_dict[n3])  # 从二阶邻居 里 选出 最大个数-一阶林觉个数
                                neb3_num = len(subgraph_list)  # 此时邻居个数设置为 第n个节点的邻居个数 加 此时邻居个数
                        else:
                            pass
                    # 循环完了 一阶邻居，找到了所有二阶邻居， 此时列表邻居个数 < 最大邻居个数
                    neb4_num = len(subgraph_list)
                    all_neb5_list = []
                    if neb4_num < self.max_neighbor:
                        for n4 in all_neb4_list:  # 一阶邻居里的每个邻居
                            nei5_li = list(set(i for i in G[n4] if i not in subgraph_list))  # 第n个二阶邻居节点的三阶邻居
                            all_neb5_list.extend(nei5_li)
                            if len(nei5_li) != 0:
                                neb_num = len(nei5_li) + neb4_num  # 总个数 = 第n个节点的邻居个数 加 此时列表邻居个数
                                if neb_num >= self.max_neighbor:  # 总个数 大于 最大邻居个数
                                    for i in nei5_li[:self.max_neighbor - neb4_num]:
                                        e_neb_dict[n4].append(i)  # 对应字典
                                    subgraph_list.extend(e_neb_dict[n4])  # 从二阶邻居 里 选出 最大个数 - 此时列表邻居个数
                                    break
                                else:
                                    for i in nei5_li:
                                        e_neb_dict[n4].append(i)
                                    subgraph_list.extend(e_neb_dict[n4])  # 从二阶邻居 里 选出 最大个数-一阶林觉个数
                                    neb4_num = len(subgraph_list)  # 此时邻居个数设置为 第n个节点的邻居个数 加 此时邻居个数
                            else:
                                pass
        if len(subgraph_list) < self.max_neighbor:
            for i in range(self.max_neighbor - len(subgraph_list)):
                e_neb_dict[subgraph_list[-1]].append(self.num_ents + i)
            subgraph_list.extend(e_neb_dict[subgraph_list[-1]])  # 从二阶邻居 里 选出 最大个数-一阶林觉个数

                        # print("all_neb4_list", all_neb2_list)
                        # print("subgraph_list4", subgraph_list)
                        # print("e_neb_dict4", e_neb_dict)

        # print("e_neb_dict", e_neb_dict)
        # print(subgraph_list)
        # print(e_neb_dict)

        neb_nebr_list = []
        for i in subgraph_list:
            if len(RG[i]) >= self.max_neb_rel:
                neb_nebr_list.append(RG[i][:self.max_neb_rel])
            else:
                if len(RG[i]) == 0:
                    for j in range(self.max_neb_rel):
                        RG[i].append(self.num_rels+j)

                else:
                    nem_neb_add_r = self.max_neb_rel - len(RG[i])
                    for j in range(nem_neb_add_r):
                        RG[i].append(RG[i][0])

                neb_nebr_list.append(RG[i])

                # print("neb_nebr_list", neb_nebr_list)
        # print(subgraph_list)
        # print(e_neb_dict)
        return subgraph_list,neb_nebr_list, e_neb_dict

    def preprocess_adj(self, adj):
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        return adj.todense()

    def get_node_sub_adj(self, triple_ids):
        batch_adj = []
        neb_eid = []
        batch_neb_nebr = []
        for hrt_id in triple_ids:
            neb_e_indices, neb_nebr, e_neb_dict = self.get_h_BFS(self.e1_neb, self.r1_neb, hrt_id)    # , r_indices
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(e_neb_dict))        # .todense()
            adj = self.preprocess_adj(adj)
            # print(adj)
            # if adj.shape[0] != 20:
            #     print(adj.shape)
            #     print(ht_id)
            batch_adj.append(adj)
            neb_eid.append(neb_e_indices)
            batch_neb_nebr.append(neb_nebr)
        batch_adj = np.array(batch_adj)
        batch_adj = Variable(torch.LongTensor(batch_adj)).to(self.device)
        return batch_adj, neb_eid, batch_neb_nebr

    def train(self):
        logging.info('START TRAINING...')
        best_mrr = 0.0
        best_batches = 0
        self.graph = self.graph_build()

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)
        for data in train_generate(self.data_file, self.batch_size, self.train_few, self.ent2id, self.rel2id, self.e1rel_e2):
            support, query, false = data
            self.batch_nums += 1

            support_adj, support_neb, support_nebr = self.get_node_sub_adj(support)
            query_adj, query_neb, query_nebr = self.get_node_sub_adj(query)
            false_adj, false_neb, false_nebr = self.get_node_sub_adj(false)

            support = Variable(torch.LongTensor(support)).to(self.device)
            query = Variable(torch.LongTensor(query)).to(self.device)
            false = Variable(torch.LongTensor(false)).to(self.device)

            support_neb = Variable(torch.LongTensor(support_neb)).to(self.device)
            query_neb = Variable(torch.LongTensor(query_neb)).to(self.device)
            false_neb = Variable(torch.LongTensor(false_neb)).to(self.device)

            support_nebr = Variable(torch.LongTensor(support_nebr)).to(self.device)
            query_nebr = Variable(torch.LongTensor(query_nebr)).to(self.device)
            false_nebr = Variable(torch.LongTensor(false_nebr)).to(self.device)

            hrt_list = [support, query, false]
            neb_list = [support_neb, query_neb, false_neb]
            nebr_list = [support_nebr, query_nebr, false_nebr]
            adj_list = [support_adj, query_adj, false_adj]
            # print(ht_list)
            # print(neb_list)
            # print(nebr_list[0].shape)
            # print(adj_list)

            positive_score, negative_score = self.fskgc(hrt_list, neb_list,nebr_list, adj_list, isEval=False)

            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            margins.append(margin_.mean().item())
            lr = adjust_learning_rate(optimizer=self.optim, epoch=self.batch_nums, lr=self.lr,
                                      warm_up_step=self.warm_up_step, max_update_step=self.max_batches)
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()
            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                logging.info(
                    'Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}, '.format(self.batch_nums, np.mean(losses), lr))
                # self.writer.write('Avg_batch_loss_every_log', np.mean(losses), self.batch_nums)

            if self.batch_nums % self.eval_every == 0:
                logging.info('Batch_nums is %d' % self.batch_nums)
                hits10, hits5, hits1, mrr = self.eval(mode='dev')   #meta=self.meta,

                # self.writer.write('HITS10', hits10, self.batch_nums)
                # self.writer.write('HITS5', hits5, self.batch_nums)
                # self.writer.write('HITS1', hits1, self.batch_nums)
                # self.writer.write('MRR', mrr, self.batch_nums)
                self.save()

                if mrr > best_mrr:
                    self.save(self.save_path + '_best')
                    best_mrr = mrr
                    best_batches = self.batch_nums
                logging.info('Best_mrr is {:.6f}, when batch num is {:d}'.format(best_mrr, best_batches))

            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info('Early stop!')
                self.save()
                break

    def eval(self, mode='dev'):
        self.fskgc.eval()
        few = self.few
        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.data_file + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.data_file + '/test_tasks.json'))

        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []
        for query_ in test_tasks.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]] for triple in support_triples]

            support_adj, support_neb, support_nebr = self.get_node_sub_adj(support_pairs)
            support_neb = Variable(torch.LongTensor(support_neb)).to(self.device)
            support_pairs = Variable(torch.LongTensor(support_pairs)).to(self.device)
            support_nebr = Variable(torch.LongTensor(support_nebr)).to(self.device)

            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]])

                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                        query_pairs.append([self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[ent]])

                query_adj, query_neb, query_nebr = self.get_node_sub_adj(query_pairs)
                query_neb = Variable(torch.LongTensor(query_neb)).to(self.device)
                query_pairs = Variable(torch.LongTensor(query_pairs)).to(self.device)
                query_nebr = Variable(torch.LongTensor(query_nebr)).to(self.device)

                ht_list = [support_pairs, query_pairs]
                neb_list = [support_neb, query_neb]
                adj_list = [support_adj, query_adj]
                nebr_list = [support_nebr, query_nebr]

                scores, _ = self.fskgc(ht_list, neb_list,  nebr_list, adj_list, isEval=True)
                scores.detach()
                scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores, kind='stable'))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f}, MRR:{:.3f}'.format(query_,
                                                                                               np.mean(hits10_),
                                                                                               np.mean(hits5_),
                                                                                               np.mean(hits1_),
                                                                                               np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of test examples {}'.format(len(candidates), len(hits10_)))
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for test')
        self.eval(mode='test')

    def eval_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for dev')
        self.eval(mode='dev')


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = read_options()
    if not os.path.exists('./logs_'):
        os.mkdir('./logs_')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    seed_everything(args.seed)

    logging.info('*' * 100)
    logging.info('*** hyper-parameters ***')
    for k, v in vars(args).items():
        logging.info(k + ': ' + str(v))
    logging.info('*' * 100)

    trainer = Trainer(args)

    if args.test:
        trainer.test_()
        trainer.eval_()
    else:
        trainer.train()
        print('last checkpoint!')
        trainer.eval_()
        trainer.test_()
        print('best checkpoint!')
        trainer.eval_(args.save_path + '_best')
        trainer.test_(args.save_path + '_best')
