#   ...................................................................................
'''
author: Palash Nandi.
'''
#   ...................................................................................

import numpy as np
import pandas as pd
import random
import pprint
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#   ...................................................................................
def read_dataset(path):
    dict_ = {}
    df = pd.read_csv(path)

    X = None
    Y = None
    
    _df = df.to_numpy()
    if 'target' in df.columns:
        X = np.array(_df[:,:-1])
        Y = np.array(_df[:,-1])
    else:
        X = np.array(_df)
    
    # print(X)
    # print(Y)

    for i in range(df.shape[0]):
        _ = {
            'id': None, 'x': None, 'y': None,
            'visit_status': 'raw', 'is_core': None, 'cluster_id':None,
            'nearest_core_point':None, 'parent_node': None
        }

        d = df.loc[i,:]

        if 'target' in df.columns:
            _['x'] = np.array(d[:-1])
            _['y'] = int(d[-1])
            df.drop(['target'], axis=1, inplace=True)
        else:
            _['x'] = np.array(d)
    
        _['id'] = i
        dict_[i] = _

    return dict_, df, X, Y


#   ...................................................................................

def is_core(radius, min_population, data, full_dataset):
    is_core = False
    selected_candidates = []
    data = np.array(data)
    full_dataset = full_dataset.to_numpy()
    
    dist_ = np.sum((full_dataset - data)**2, axis=1)
    # dist_ = np.sqrt(dist_)
    print(f'\t\tdistances: {dist_}')
    
    for pos_i,d in enumerate(dist_):
        if d <= radius:
            print(f'\t\tfor dist: {d}, radius: {radius} => selected')
            selected_candidates.append(pos_i)

    print(f'\t\tselected_candidates: {len(selected_candidates)}, min_population: {min_population} => is_core: {is_core}')
    if len(selected_candidates) >= min_population:
        is_core = True
        

    return {
    'is_core': is_core,
    'direct_reachable': selected_candidates
    }

#   ...................................................................................

def update_context(waiting_queue, explored_queue, direct_reachable_candidates, data_dict, dataset_dict):
    for i in explored_queue:
        try:
            direct_reachable_candidates.remove(i)
        except:
            pass

    waiting_queue += direct_reachable_candidates
    
    for i in direct_reachable_candidates:
        dataset_dict[i]['visit_status'] = 'touched'
        dataset_dict[i]['cluster_id'] = data_dict['cluster_id']
        dataset_dict[i]['parent_node'] = data_dict['id']
        dataset_dict[i]['nearest_core_point'] = data_dict['id']
    
#   ...................................................................................

def calculate(cluster_id, available_nodes, dataset_dict, data_df):
    radius = 15
    min_population = 3
    
    available_nodes = list(set(available_nodes))
    holding_point = random.choice(available_nodes)
    waiting_queue = [holding_point]
    explored_queue = []
    
    
    while len(waiting_queue) != 0:
        waiting_queue = list(set(waiting_queue))

        print(f'\nwaiting_queue: {waiting_queue}')
        holding_point = waiting_queue.pop(0)
        print(f'holding_point: {holding_point}')
        print(f'waiting_queue: {waiting_queue}\n')

        available_nodes.remove(holding_point)

        data_dict = dataset_dict[holding_point]
        
        print(f"For: {data_dict['id']}, {data_dict['x']}")
        if data_dict['id'] in explored_queue:
            continue
        
        if data_dict['visit_status'] == 'raw':
            print(f"\tRaw:_{data_dict['id']}: raw")
            data_dict['cluster_id'] = cluster_id
            data_dict['visit_status'] = 'explored'
            explored_queue.append(data_dict['id'])
            
            _dict_= is_core(radius, min_population, data_dict['x'], data_df)
            
            data_dict['is_core'] = _dict_['is_core']
            direct_reachable_candidates = _dict_['direct_reachable']
            if data_dict['is_core'] == True:
                data_dict['nearest_core_point'] = data_dict['id']
            
            # print(data_dict)
            print(f'\tRaw:_direct_reachable_candidates: {len(direct_reachable_candidates)}')
            print(f'\tRaw:_direct_reachable_candidates: {direct_reachable_candidates}')
            
            update_context(waiting_queue, explored_queue, direct_reachable_candidates, data_dict, dataset_dict)
            print(f'\tRaw:_explored_queue: {explored_queue}')
            print(f'\tRaw:_info: {data_dict}')
            # print(f'waiting_queue : {waiting_queue}')

            # for w_i in waiting_queue:
            #     print(dataset_dict[w_i])
            continue
            # break

        if data_dict['visit_status'] == 'touched':
            print(f"\tTouched:_{data_dict['id']}: {data_dict['visit_status']}")
            data_dict['visit_status'] = 'explored'
            explored_queue.append(data_dict['id'])
            print(f"\tTouched:_{data_dict['id']}: {data_dict['visit_status']}")
            
            # print(data_dict)
            # print(waiting_queue)
            
            _dict_= is_core(radius, min_population, data_dict['x'], data_df)
            
            data_dict['is_core'] = _dict_['is_core']
            direct_reachable_candidates = _dict_['direct_reachable']
            if data_dict['is_core'] == True:
                data_dict['nearest_core_point'] = data_dict['id']
            
            # print(data_dict)
            print(f'\tTouched:_ direct_reachable_candidates: {len(direct_reachable_candidates)}')
            print(f'\tTouched:_direct_reachable_candidates: {direct_reachable_candidates}')
            
            update_context(waiting_queue, explored_queue, direct_reachable_candidates, data_dict, dataset_dict)
            print(f'\tTouched:_explored_queue: {explored_queue}')
            print(f'\tTouched:_info: {data_dict}')
            # print(f'waiting_queue : {waiting_queue}')
            continue

        
        # visited_queue.append(data_dict['id'])
        # isCore(data, df)

        # break
    return available_nodes

#   ...................................................................................

def display_clusters(dataset_dict):
    id_2_cluster_id = {}
    cluster_id_2_nodes = {}

    for i, dict_ in dataset_dict.items():
        _id = i
        _cid = dict_['cluster_id']

        id_2_cluster_id[_id] = _cid
        try:
            _ = cluster_id_2_nodes[_cid] 
            _.append(_id)
            cluster_id_2_nodes[_cid] = _
        except:
            cluster_id_2_nodes[_cid] = [_id]
    
    for k,v in cluster_id_2_nodes.items():
        print(f'{k}: {v} i.e. {len(v)} nodes.')

    return id_2_cluster_id, cluster_id_2_nodes

#   .............................From scratch......................................................

# path = '/content/iris_full.csv'
path = '//home/palash/ML_GitHub/DBSCAN/customed_data.csv'
dataset_dict, data_df, X, Y = read_dataset(path)
cluster_id = -1
available_nodes = list(dataset_dict.keys())

while len(available_nodes)> 0:
    cluster_id += 1
    print('...'*50)
    available_nodes = calculate(cluster_id, available_nodes, dataset_dict, data_df)
    print(f'\n####..... available_nodes: {len(available_nodes)}\n')
    
print(f'Clustering result from DBSCAN from scratch')
id_2_cluster_id, cluster_id_2_nodes = display_clusters(dataset_dict)

#   .............................From sklearn.cluster's DBSCAN......................................................

print(f"\nClustering result from sklearn.cluster's DBSCAN")

c_id_to_node_id = {}

dbscan_clustering = DBSCAN(eps=3, min_samples=7).fit(X)
labels = dbscan_clustering.labels_

for node_id, c_id in enumerate(labels):
    try:
        _ = c_id_to_node_id[c_id] 
        _.append(node_id)
        c_id_to_node_id[c_id] = _
    except:
        c_id_to_node_id[c_id] = [node_id]

for c_id, nodes in c_id_to_node_id.items():
    print(f'{c_id}: {nodes} i.e. {len(nodes)} nodes.')
