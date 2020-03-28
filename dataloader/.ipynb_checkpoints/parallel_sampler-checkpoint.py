import time
import datetime
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np
from pytorch_transformers import BertModel

import dataloader.utils as utils


class ParallelSampler():
    def __init__(self, data,args,num_episodes=None):
        self.data = data
        self.num_episodes = num_episodes
        self.args = args
        
        if args.dataset == 'event':
            self.all_classes = np.unique(self.data['category'])
        else:
            self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        if args.dataset == 'event':
            for y in self.all_classes:
                self.idx_list.append(
                        np.squeeze(np.argwhere(self.data['category'] == y))) 
        else:
            for y in self.all_classes:
                self.idx_list.append(
                        np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers
        
        print("{}, Initializing parallel data loader with {} processes".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            self.num_cores), flush=True)

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                    Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(
                    self.num_classes)[:self.args.way]

            source_classes = []
            for j in range(self.num_classes):
                if j not in sampled_classes:
                    source_classes.append(self.all_classes[j])
            source_classes = sorted(source_classes)

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(self.idx_list[y].size)
                support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:self.args.shot+self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)

            # aggregate examples
            max_support_len = np.max(self.data['sentences_lengths'][support_idx])
            max_query_len = np.max(self.data['sentences_lengths'][query_idx])

            support = utils.select_subset(
                self.data, {}, ['input_ids','input_mask','sentences_lengths',
                                'token_type_ids','label'],support_idx, max_support_len)
            query = utils.select_subset(
                self.data, {},['input_ids','input_mask','sentences_lengths',
                 'token_type_ids','label'],query_idx, max_query_len)
            
            done_queue.put((support, query))


    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue
