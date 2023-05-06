import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import pickle
import os
from typing import List


class TriDataSampler(Sampler):
    """
    Data Sampler for recommender systems

    Args:
        labels: a 2-D array: [item_num * task_num]
        batchSize: number of all items in a batch = multi_task * (posNum + negNum)
        posNum: number of positive items for each label in a batch
        multi_task: number of labels in a batch
    """
    def __init__(self, labels, batchSize, posNum, multi_task):
        self.labels = np.array(labels)
        self.multi_task = multi_task
        self.batchSize = batchSize
        
        self.posNum = posNum
        self.negNum = self.batchSize//multi_task - self.posNum

        self.label_dict = {}
        for i in range(self.labels.shape[1]):         # iterate over all labels
            task_label = self.labels[:,i]
            pos_index = np.flatnonzero(task_label>0)  # indices for items with postive labels
            ###To avoid sampling error
            while len(pos_index) < self.posNum: 
                pos_index = np.concatenate((pos_index,pos_index))
            np.random.shuffle(pos_index)

            neg_index = np.flatnonzero(task_label==0) # indices for items with postive labels
            while len(neg_index) < self.negNum: 
                neg_index = np.concatenate((neg_index,neg_index))
            np.random.shuffle(neg_index)

            self.label_dict.update({i:(pos_index,neg_index)})

        self.posPtr, self.negPtr = np.zeros(self.labels.shape[1], dtype=np.int64), np.zeros(self.labels.shape[1], dtype=np.int64)
        self.taskPtr, self.tasks = 0, np.random.permutation(list(range(self.labels.shape[1])))

        self.batchNum = self.labels.shape[1] // self.multi_task

        self.ret_labels = np.empty(self.batchNum*self.multi_task*(self.posNum+self.negNum), dtype=np.int64) # store the pos and neg item ids for all batchNum batches 


    def __iter__(self):

        beg = 0 # beg is the pointer for self.ret

        for batch_id in range(self.batchNum):
            task_ids = self.tasks[self.taskPtr:self.taskPtr+self.multi_task]  # randomly sample task_ids (number: self.multi_task)
            self.taskPtr += self.multi_task
            if self.taskPtr >= len(self.tasks):
                np.random.shuffle(self.tasks)            
                self.taskPtr = self.taskPtr % len(self.tasks)                 # if reach the end, then shuffle the list and mod the pointer
                                   
            for task_id in task_ids:
                item_list = np.empty(self.posNum+self.negNum, dtype=np.int64)

                if self.posPtr[task_id]+self.posNum > len(self.label_dict[task_id][0]):
                    temp = self.label_dict[task_id][0][self.posPtr[task_id]:]
                    np.random.shuffle(self.label_dict[task_id][0])
                    self.posPtr[task_id] = (self.posPtr[task_id]+self.posNum)%len(self.label_dict[task_id][0])
                    if self.posPtr[task_id]+len(temp) < self.posNum:
                        self.posPtr[task_id] += self.posNum-len(temp)
                    item_ids = np.concatenate((temp,self.label_dict[task_id][0][:self.posPtr[task_id]]))
                    item_list[:self.posNum] = item_ids
                else:
                    item_ids = self.label_dict[task_id][0][self.posPtr[task_id]:self.posPtr[task_id]+self.posNum]
                    item_list[:self.posNum] = item_ids
                    self.posPtr[task_id] += self.posNum

                if self.negPtr[task_id]+self.negNum > len(self.label_dict[task_id][1]):
                    temp = self.label_dict[task_id][1][self.negPtr[task_id]:]
                    np.random.shuffle(self.label_dict[task_id][1])
                    self.negPtr[task_id] = (self.negPtr[task_id]+self.negNum)%len(self.label_dict[task_id][1])
                    if self.negPtr[task_id]+len(temp) < self.negNum:
                        self.negPtr[task_id] += self.negNum-len(temp)
                    item_ids = np.concatenate((temp,self.label_dict[task_id][1][:self.negPtr[task_id]]))
                    item_list[self.posNum:] = item_ids
                else:
                    item_ids = self.label_dict[task_id][1][self.negPtr[task_id]:self.negPtr[task_id]+self.negNum]
                    item_list[self.posNum:] = item_ids
                    self.negPtr[task_id] += self.negNum                        # sample negNum negative items for task_id
                
                self.ret_labels[beg:beg+self.posNum+self.negNum] = item_list
                beg += self.posNum+self.negNum

        return iter(self.ret_labels)


    def __len__ (self):
        return len(self.ret_labels)


    
