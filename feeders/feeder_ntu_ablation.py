import numpy as np

from torch.utils.data import Dataset
import h5py
from feeders import tools
import feeders.augmentations as augmentations

unseen_list_ntu60 = {'1':[50, 40, 30, 37, 12, 48, 45, 49, 8, 29, 58, 13, 1, 39, 27, 47, 14, 52, 3, 44],
                    '2': [41, 21, 52,  6, 12, 36, 24, 56, 35, 57, 15, 26, 39, 53, 19,  4, 27, 25, 17, 47],
                    '3': [46, 10, 47, 39, 55, 14, 58, 53, 13, 40, 24,  9, 45, 23, 27,  3,  7, 54, 33, 17],
                    '4': [21, 55, 11, 43, 41,  3, 52, 39, 46, 59, 47, 15, 17, 54, 40, 33,  9, 38, 31, 57],
                    '5': [56, 14, 17,  7, 40, 52, 37, 50, 36,  6, 44, 11, 41,  9, 47, 24, 53,  2, 10, 58]}

seen_list_ntu60_10seen =  {'1':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    '2': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    '3': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                    '4': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                    '5': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]}
def get_mapping(run):
    labels = [j for j in range(60)]
    label_mapping = []
    for i in labels:
        if i not in unseen_list_ntu60[str(run)]:
            label_mapping.append(i)
    return label_mapping

def get_mapping_seen(run):
    labels = [j for j in range(60)]
    label_mapping = []
    for i in labels:
        if i in seen_list_ntu60_10seen[str(run)]:
            label_mapping.append(i)
    return label_mapping
class Feeder(Dataset):
    def __init__(self, data_path, run=1, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.label_mapping = get_mapping_seen(run) 
        self.run = run
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()
    def load_data(self):
        # data: N C V T M
        h5 = h5py.File(self.data_path,'r')
        #npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = []
            self.label = []
            self.data_raw = np.array(h5.get('x'))
            self.label_raw = np.array(h5.get('y'))
            self.label_raw = np.where(self.label_raw > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                #print(label)
                if label in seen_list_ntu60_10seen[str(self.run)]:
                    self.data.append(data)
                    label = self.label_mapping.index(label)
                    self.label.append(label)
                    self.sample_name.append(name)            
        elif self.split == 'test_seen' or self.split == 'test_unseen':
            self.data_raw = np.array(h5.get('test_x'))
            self.data = []
            self.label_raw = np.array(h5.get('test_y'))
            self.label_raw = np.where( self.label_raw> 0)[1]
            self.label = []
            self.sample_name_raw = ['test_' + str(i) for i in range(len(self.data_raw))]
            self.sample_name = []
            if self.split == 'test_seen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label in seen_list_ntu60_10seen[str(self.run)]:
                        self.data.append(data)
                        label = self.label_mapping.index(label)
                        self.label.append(label)
                        self.sample_name.append(name)
            elif self.split == 'test_unseen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label not in seen_list_ntu60_10seen[str(self.run)]:
                        self.data.append(data)
                        label = 11
                        self.label.append(label)
                        self.sample_name.append(name)                
        else:
            raise NotImplementedError('data split only supports train/test')
        self.data = np.stack(self.data, 0)
        self.label = np.stack(self.label, 0)
        N, T, _ = self.data.shape

        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    def __len__(self):
        return len(self.label)
    def __iter__(self):
        return self
        
    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        #if self.random_rot:
        #data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        data_numpy_v1 = data_numpy
        # randomly select  one of the spatial augmentations 
        data_numpy_v1 = tools.random_rot(data_numpy_v1)
        import random
        flip_prob  = random.random()
        if flip_prob < 0.5:
            data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1)
        else:
            data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1)

        data_numpy = tools.random_rot(data_numpy)
        from .bone_pairs import ntu_pairs
        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in ntu_pairs:
            bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        vel_data_numpy = np.zeros_like(data_numpy)
        vel_data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        vel_data_numpy[:, -1] = 0

        
        bone_data_numpy_v1 = np.zeros_like(data_numpy_v1)
        for v1, v2 in ntu_pairs:
            bone_data_numpy_v1[:, :, v1 - 1] = data_numpy_v1[:, :, v1 - 1] - data_numpy_v1[:, :, v2 - 1]
        vel_data_numpy_v1 = np.zeros_like(data_numpy_v1)
        vel_data_numpy_v1[:, :-1] = data_numpy_v1[:, 1:] - data_numpy_v1[:, :-1]
        vel_data_numpy_v1[:, -1] = 0

        #data_numpy_lower = data_numpy[:,::2]
        #data_numpy_upper = data_numpy[:,::4]
        if self.split == 'train':
            return [data_numpy, bone_data_numpy, vel_data_numpy], label, index
        else:
            return [data_numpy, bone_data_numpy, vel_data_numpy], label, index
        

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
