import numpy as np

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range = np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        
        self.local_sum = np.zeros(self.size, dtype=np.float32)
        self.local_sum_sq = np.zeros(self.size, dtype = np.float32)
        self.local_cnt = np.zeros(1, dtype = np.float32)
        
        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype = np.float32)
        self.running_sum = np.zeros(self.size, dtype = np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype = np.float32)
        self.running_cnt = 1
        
    def update_local_stats(self, new_data):
        self.local_sum += new_data.sum(axis=0)
        self.local_sum_sq += (np.square(new_data)).sum(axis = 0)
        self.local_cnt += new_data.shape[0]
        
    
        
    def normalize_observation(self, v):
        clip_range = self.default_clip_range
        return np.clip((v-self.running_mean)/ self.running_std,
                       -clip_range, clip_range).astype(np.float32)
        
    def recompute_global_stats(self):
        self.running_cnt += self.local_cnt
        self.running_sum += self.local_sum
        self.running_sum_sq += self.local_sum_sq
        
        self.local_cnt[...] = 0
        self.local_sum[...] = 0
        self.local_sum_sq[...] = 0 
        
        self.running_mean = self.running_mean/self.running_cnt
        tmp  =self.running_sum_sq/self.running_cnt - np.square(self.running_sum/self.running_cnt)
        self.running_std = np.sqrt(np.maximum(np.square(self.eps), tmp))         
        