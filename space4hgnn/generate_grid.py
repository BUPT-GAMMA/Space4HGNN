import yaml
import argparse
import os
hidden_dim = [8, 16]
layers_pre_mp = [1]
layers_post_mp = [1, 2]
layers_gnn = [1, 2, 3]
stage_type = ['skipconcat']
activation = ['tanh']
has_bn = [True]
has_l2norm = [True, False]
mini_batch_flag = [False]
macro_func = ['sum']
dropout = [0, 0.2]
lr = [0.01]
weight_decay = 0.0001
patience = 40
optimizer = ['Adam']
# TODO : change
max_epoch = [50, 100]
num_heads = [2]
#max_epoch = 2

# hidden_dim = [64]
# layers_pre_mp = [1, 2]
# layers_post_mp = [1, 2]
# layers_gnn = [2, 4]
# stage_type = ['skipsum','skipconcat']
# activation = ['elu']
# has_bn = [True]
# has_l2norm = [True, False]
# mini_batch_flag = [False]
# macro_func = ['mean']
# dropout = [0.0, 0.3]
# lr = [0.01]
# weight_decay = 0.0001
# patience = 40
# # TODO : change
# max_epoch = [400]

def getIndex():
    dicts = {}

    count = 0
    for a in range(len(layers_pre_mp)):
        dict = {}
        dict['layers_pre_mp'] = a
        for b in range(len(layers_pre_mp)):
            dict['layers_post_mp'] = b
            for c in range(len(layers_gnn)):
                dict['layers_gnn'] = c
                for d in range(len(stage_type)):
                    dict['stage_type'] = d
                    for e in range(len(activation)):
                        dict['activation'] = e
                        for f in range(len(has_bn)):
                            dict['has_bn'] = f
                            for g in range(len(has_l2norm)):
                                dict['has_l2norm'] = g
                                for h in range(len(macro_func)):
                                    dict['macro_func'] = h
                                    for i in range(len(dropout)):
                                        dict['dropout'] = i
                                        for j in range(len(lr)):
                                            dict['lr'] = j
                                            for k in range(len(max_epoch)):
                                                dict['max_epoch'] = k
                                                dicts[count] = dict.copy()
                                                count = count + 1
    return dicts


def makeDict(aggr, index):
    dict = {
        'hidden_dim': 64,
        'layers_pre_mp': layers_pre_mp[index['layers_pre_mp']],
        'layers_post_mp': layers_post_mp[index['layers_post_mp']],
        'layers_gnn': layers_gnn[index['layers_gnn']],
        'stage_type': stage_type[index['stage_type']],
        'activation': activation[index['activation']],
        'dropout': dropout[index['dropout']],
        'has_bn': has_bn[index['has_bn']],

        'has_l2norm': has_l2norm[index['has_l2norm']],
        'lr': lr[index['lr']],
        'weight_decay': 0.0001,
        'patience': 40,
        'max_epoch': max_epoch[index['max_epoch']],
        'mini_batch_flag': False,
        'macro_func': macro_func[index['macro_func']],
        'gnn_type': aggr,
    }
    # dict[key] = value
    return dict

def generate(aggr, configfile):
    datasets = ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBn-Freebase']
    datasets2 = ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']
    models = ['homo_GNN', 'relation_HGNN', 'mp_GNN']
    dicts = {}
    dicts2 = {}

    indexes = getIndex()

    size = len(indexes)
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    if not os.path.exists('./space4hgnn/config/{}'.format(configfile)):
        os.makedirs('./space4hgnn/config/{}'.format(configfile))

    for count in range(size):
        index = indexes[count]
        for a in datasets:
            dict = {}
            for b in models:
                dict[b] = makeDict(aggr, index)
            dicts[a] = dict
        for a in datasets2:
            dict = {}
            for b in models:
                dict[b] = makeDict(aggr, index)
            dicts2[a] = dict
        aproject = {'node_classification': dicts,
                    'link_prediction': dicts2
                    }
        name = 'config/{}/'.format(configfile) + str(count) + '.yaml'
        yamlPath = os.path.join(fileNamePath, name)

        f = open(yamlPath,'w')
        print(yaml.dump(aproject,f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggr', '-a', default='gcnconv', type=str, help='gnn type')
    parser.add_argument('--configfile', '-c', default='config', type=str, help='config file path')
    args = parser.parse_args()
    generate(args.aggr, args.configfile)