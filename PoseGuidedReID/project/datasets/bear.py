
# encoding: utf-8
"""
@author:  Mu
"""

import pandas as pd
import os.path as osp
from .bases import BaseDataset
import random


class Bear(BaseDataset):
    """
    Bear
    bear data 

    Dataset statistics:
    # identities: 
    """

    def __init__(self, 
                 cfg=None,
                 root='', 
                 verbose=True, 
                 img_dir='',
                 **kwargs):
        super(Bear, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.img_dir = img_dir
        self.data_dir = self.dataset_dir
        self.cfg = cfg
        
        self.sub_train_year = self.cfg.DATASETS.SUB_TRAIN_YEAR
        self.is_body_image = self.cfg.DATASETS.BODY_INPUT
        self.update_train_csv = self.cfg.DATASETS.UPDATE_TRAIN_CSV

        self.train_dir = osp.join(self.data_dir, 'train_iid.csv')
        
        if self.update_train_csv is not None:
            self.train_dir = osp.join(self.data_dir, self.update_train_csv)

        self.test_iid_dir = osp.join(self.data_dir, 'test_iid.csv')
        self.test_ood_dir = osp.join(self.data_dir, 'test_ood.csv')
        self.val_iid_dir = osp.join(self.data_dir, 'val_iid.csv')
        self.all_5year_dir = osp.join(self.data_dir, '../full_data.csv')
        self.sex_csv = osp.join(self.img_dir, 'annotation', 'sex.csv')
        self.df_sex = pd.read_csv(self.sex_csv)
        self.shuffle_ids = self.cfg.DATASETS.IDWISE_PERT
        self.shuffle_images = self.cfg.DATASETS.IMAGEWISE_PERT

        self._check_before_run()

        self.pid_container = {'train':{}, 'iid':{}, 'ood':{}, 'val':{}}
        self.cid_container = set()
        self.view_container = set()
        self.bg_container = set()
        self.time_container = set()
        self.year_container = set()
        
        ### NOTE: this info is fixed -- becuase we have only 7 cameras in the dataset (2017-2022)
        self.camera2id = {'DC-G9': 0, 'Canon EOS 77D': 1, 'Canon EOS 7D Mark II': 2, 'DSC-RX10M4': 3, 'Canon EOS REBEL T2i': 4, 'NIKON D3300': 5, 'NIKON D810': 6} 

        self.train, self.train_year_list, self.n_train_classes = self.process_dir(self.train_dir, data_type='train')
        self.test_iid, self.test_iid_year_list, self.n_test_iid_classes = self.process_dir(self.test_iid_dir, data_type='iid')
        self.test, self.test_year_list, self.n_test_classes  = self.process_dir(self.test_ood_dir, data_type='ood')
        self.val, self.val_year_list, self.n_val_classes = self.process_dir(self.val_iid_dir, data_type='val')
        # self.all_5year, _ = self.process_dir(self.all_5year_dir)

        if verbose:
            print("=> Bear data loaded")
            self.print_dataset_statistics()


    def process_dir(self, dir_path, data_type=None):
        print('Processing dir: ', dir_path)
        # data = pd.read_csv(dir_path)
        data = self.load_data(dir_path)
        
        ### if self.sub_train_year is not empty, then only use the data from the specified year
        if len(self.sub_train_year) > 0 and 'ood' not in dir_path:
            data = data[data['year'].isin(self.sub_train_year)]

        img_paths = data['image'].tolist()
        if self.is_body_image:
            img_paths = data['body_image'].tolist()
        
        # pid_list = data['bearname_id'].tolist()
        camera_id_list = data['camera'].apply(lambda x: self.camera2id[x]).tolist()
        
        
        ## ### shuffle the data['id'] to test the annotation performance
        if self.shuffle_ids is not None or self.shuffle_images is not None:
            ### shuffle rules (shuffle same ID to a wrong one + shuffle the different images from same ID)
            ### type 1: IDwise_perturb - find the the n% of IDs to shuffle to the same wrong ID 
            ### type 2: Imagewise_perturb - find random k% samples (image) to a wrong ID
            ### type 3: IDwise_perturb + Imagewise_perturb
            
            if 'train_iid' in dir_path:
                print(f"start shuffling {self.shuffle_ids}% IDs in Training set, and {self.shuffle_images}% images in Training set")
                random.seed(32)
                # get the number of IDs to shuffle
                unique_id_list = data['id'].unique().tolist()
                id_list = data['id'].tolist()
                
                if int(self.shuffle_ids) == 100 or int(self.shuffle_images) == 100:
                    # complete messy data -- random shuffle id_list
                    random.shuffle(id_list)
                else:
                    # type 1: IDwise_perturb  - map the random_ids_to_shuffle to other_ids -- make sure the same ID is shuffled to the another wrong ID
                    if int(self.shuffle_ids) != 0:
                        num_ids = int(len(data['id'].unique()) * int(self.shuffle_ids) / 100)
                        # get random num_ids from the unique_id_list
                        random_ids_to_shuffle = random.sample(unique_id_list, num_ids)
                        # check id_list, if the id is in random_ids_to_shuffle, then shuffle it
                        shuffle_map = {}
                        for i in range(len(random_ids_to_shuffle)):
                            other_ids = unique_id_list.copy() 
                            other_ids.remove(random_ids_to_shuffle[i])
                            other_id = random.choice(other_ids)
                            shuffle_map[random_ids_to_shuffle[i]] = other_id
                        for i in range(len(id_list)):
                            if id_list[i] in random_ids_to_shuffle:
                                id_list[i] = shuffle_map[id_list[i]]
                            
                    # type 2: Imagewise_perturb - shuffle the k-% images to a wrong ID
                    num_images = int(len(img_paths) * int(self.shuffle_images) / 100)
                    random_images = random.sample(img_paths, num_images)
                    for i in range(len(id_list)):
                        if img_paths[i] in random_images:
                            id_list[i] = random.choice(unique_id_list)

                data['id'] = id_list

        bear_name_list = data['id'].tolist()
        
        ### turn data['id'] names -> data['pid'] index -> to make it compatible with the original code
        # Get unique names and sort them (optional)
        unique_names = sorted(data['id'].unique())
        name_to_pid = {name: idx for idx, name in enumerate(unique_names)}
        data['pid'] = data['id'].apply(lambda x: name_to_pid[x])
        pid_list = data['pid'].tolist()

        ### 
        year_list = data['year'].tolist()
        if 'is_unique' not in data.columns:
            data['is_unique'] = '0'
        is_unique_id = data['is_unique'].tolist()
        timestamps = data['timestamp'].tolist()

        self.pid_sex_container = set()
        classes = set()
        
        ### 

        data = []
        for i, (_img_path, pid, year, cid, bear_name) in enumerate(zip(img_paths, pid_list, year_list, camera_id_list, bear_name_list)):
            if self.is_body_image:
                img_path = osp.join(self.img_dir, _img_path)
            else:
                img_path = osp.join(self.img_dir, str(year), _img_path)
            self.pid_container[data_type][pid]=bear_name
            self.cid_container.add(cid)
            self.year_container.add(year)
            classes.add(pid)
            sex_id = -1

            meta_info = [pid, bear_name, cid, timestamps[i], is_unique_id[i], _img_path, sex_id]
            data.append((img_path, pid, cid, meta_info))

            self.pid_sex_container.add((bear_name, sex_id))

        num_classes = len(classes)

        return data, set(year_list), num_classes

