import json
import random
from state_reward import StateReward
from quarto_utils import checkState

#This class handles the pre-processing of the datasets stored in dataset/raw



class PreProcessDataset:
    MAX_SIZE = 1000
    PROCESSED_TARGET = "dataset/pre_processed/training_dataset_v2.json"
    PROCESSED_TARGET_2 = "dataset/pre_processed/validation_dataset_v2.json"
    def count_state_size(state):
        c = 0
        for i in state: 
            if i>-1: c+=1;
        return c
    def pre_process(paths):
        dataset = None
        processed = dict()
        sum_ = 0
        for path in paths:
            with open(path, 'r') as source:
                dataset = json.load(source)
            for record in dataset['exp']:
                full, winning = checkState(record[0][0])
                count = PreProcessDataset.count_state_size(record[0][0])
                if (full or winning or count==15):
                    continue 
                if not count in processed:
                    processed[count] = {'pos': [], 'neg': [], 'neut':[]}
                
                if record[1] > 0:
                    processed[count]['pos'].append(StateReward.process_state_for_dataset(record))
                elif record[1] < 0:
                    processed[count]['neg'].append(StateReward.process_state_for_dataset(record))
                else:
                    processed[count]['neut'].append(StateReward.process_state_for_dataset(record))
                sum_ += 1
        
        #Balance
        sum_ = 0
        print("Balanced 1")
        for key in processed.keys():
            target = max(len(processed[key]['neg']), len(processed[key]['neut']), 30)
            target = min(target, PreProcessDataset.MAX_SIZE)
            for t in processed[key].keys():
                if (len(processed[key][t]) > target):
                    processed[key][t] = random.sample(processed[key][t], target)
                    sum_ += target 
                else:
                    sum_ += len(processed[key][t])
                random.shuffle(processed[key][t])
        for key in processed.keys():
            print(f"{key}: {len(processed[key]['pos'])} pos, {len(processed[key]['neut'])} neut, {len(processed[key]['neg'])} neg")
        print(f"In total: {sum_} states available to export")
        #Get two separate datasets
        tranining_dataset = dict()
        validate_dataset = dict()
        for key in processed.keys():
            tranining_dataset[key] = []
            validate_dataset[key] = []
            for t in processed[key].keys():
                len_ = int(len( processed[key][t])/2)
                if (len_ > 50):
                    tranining_dataset[key].extend( processed[key][t][:len_] )
                    validate_dataset[key].extend( processed[key][t][len_:] )
                else:
                    tranining_dataset[key].extend( processed[key][t] )
                    validate_dataset[key].extend( processed[key][t] )
        #Save dataset
        with open(PreProcessDataset.PROCESSED_TARGET, 'w') as dataset:
            dataset.write(json.dumps(tranining_dataset))
        with open(PreProcessDataset.PROCESSED_TARGET_2, 'w') as dataset:
            dataset.write(json.dumps(validate_dataset))


    


PreProcessDataset.pre_process([
    "dataset/raw/dataset_v0__0.json",
    "dataset/raw/dataset_v1__0.json",
    "dataset/raw/dataset_v1__1.json",
    "dataset/raw/dataset_v1__2.json",
    "dataset/raw/dataset_v1__3.json",
    "dataset/raw/dataset_v1__4.json",
    "dataset/raw/dataset_v3__0.json",
    "dataset/raw/dataset_v3__1.json",
    "dataset/raw/dataset_v3__2.json",
    "dataset/raw/dataset_v3__3.json",
    "dataset/raw/dataset_v3__4.json",
    "dataset/raw/dataset_v4__0.json",
    "dataset/raw/dataset_v4__1.json",
    "dataset/raw/dataset_v4__2.json",
    "dataset/raw/dataset_v4__3.json",
    "dataset/raw/dataset_v4__4.json",
    "dataset/raw/dataset_v5__0.json",
    "dataset/raw/dataset_v5__1.json",
    "dataset/raw/dataset_v5__2.json",
    "dataset/raw/dataset_v5__3.json",
    "dataset/raw/dataset_v5__4.json",
    "dataset/raw/dataset_v6__0.json",
    "dataset/raw/dataset_v6__1.json",
    "dataset/raw/dataset_v6__2.json",
    "dataset/raw/dataset_v7__0.json",
    "dataset/raw/dataset_v8__0.json",
    "dataset/raw/dataset_v8__1.json",
    "dataset/raw/dataset_v8__2.json",
    "dataset/raw/dataset_v8__3.json",
    "dataset/raw/dataset_v8__4.json",
    "dataset/raw/dataset_v9__0.json",
    "dataset/raw/dataset_v9__1.json",
    "dataset/raw/dataset_v9__2.json",
    "dataset/raw/dataset_v9__3.json",
    "dataset/raw/dataset_v10__0.json",
    "dataset/raw/dataset_v10__1.json",
    "dataset/raw/dataset_v10__2.json",
    "dataset/raw/dataset_v11__0.json",
    "dataset/raw/dataset_v11__1.json",
    "dataset/raw/dataset_v11__2.json"
])