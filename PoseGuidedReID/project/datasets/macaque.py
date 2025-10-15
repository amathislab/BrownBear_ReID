
# encoding: utf-8

import pandas as pd
import os.path as osp
from torch.utils.data import Dataset


class Macaque(Dataset):
    """
    Macaque data
    Dataset statistics:
    # identities: 
    """

    def __init__(self, 
                 cfg=None,
                 root='', 
                 verbose=True, 
                 img_dir='',
                 **kwargs):
        super(Macaque, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.img_dir = img_dir
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'train_iid.csv')
        self.test_iid_dir = osp.join(self.data_dir, 'test_iid.csv')
        self.test_ood_dir = osp.join(self.data_dir, 'test_ood.csv')
        self.val_iid_dir = osp.join(self.data_dir, 'val_iid.csv')

        self._check_before_run()

        self.pid_container = {'train':{}, 'iid':{}, 'ood':{}, 'val':{}}
        self.cid_container = set()
        self.view_container = set()
        self.bg_container = set()
        self.time_container = set()
        self.year_container = set()

        self.train, self.train_year_list, self.n_train_classes = self.process_dir(self.train_dir, data_type='train')
        self.test_iid, self.test_iid_year_list, self.n_test_iid_classes = self.process_dir(self.test_iid_dir, data_type='iid')
        self.test, self.test_year_list, self.n_test_classes  = self.process_dir(self.test_ood_dir, data_type='ood')
        self.val, self.val_year_list, self.n_val_classes = self.process_dir(self.val_iid_dir, data_type='val')

        if verbose:
            print("=> Bear data loaded")
            self.print_dataset_statistics()

        # self.num_train_pids, self.num_train_cams, self.num_train_imgs = self.get_imagedata_info(self.train)

    def _check_before_run(self):
        """Check if folder is available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    
    def get_imagedata_info(self, data):
        pids, cams = [], []
        for img_path, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        return len(pids), len(cams), len(data)

    def print_dataset_statistics(self):

        print("Dataset statistics:")
        print("  --------------------------------------------------------------------")
        print("  subset    | year(min-max) | #ids| #cameras| #images ")
        print("  --------------------------------------------------------------------")

        total_years = f"{min(self.year_container)}-{max(self.year_container)}"
        train_years = f"{min(self.train_year_list)}-{max(self.train_year_list)}"
        test_iid_years = f"{min(self.test_iid_year_list)}-{max(self.test_iid_year_list)}"
        test_ood_years = f"{min(self.test_year_list)}-{max(self.test_year_list)}"
        val_iid_years = f"{min(self.val_year_list)}-{max(self.val_year_list)}"

        train_info = "  |  ".join(map(str, self.get_imagedata_info(self.train)))
        test_iid_info = "  |  ".join(map(str, self.get_imagedata_info(self.test_iid)))
        test_ood_info = "  |  ".join(map(str, self.get_imagedata_info(self.test)))
        val_iid_info = "  |  ".join(map(str, self.get_imagedata_info(self.val)))


        print(f"  {'train':<9} | {train_years:<13} | {train_info}")
        print(f"  {'test_iid':<9} | {test_iid_years:<13} | {test_iid_info}")
        print(f"  {'test_ood':<9} | {test_ood_years:<13} | {test_ood_info}")
        print(f"  {'val_iid':<9} | {val_iid_years:<13} | {val_iid_info}")
        print("  --------------------------------------------------------------------")

    def process_dir(self, dir_path, data_type=None):
        print('Processing dir: ', dir_path)
        data = pd.read_csv(dir_path)

        img_paths = data['image'].tolist()
        camera_id = [0] * len(img_paths) #data['camera_id'].tolist() -> useless info, not used in training
        
        name_list = data['id'].tolist()
        
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

        data = []
        for i, (_img_path, pid, year, cid, name) in enumerate(zip(img_paths, pid_list, year_list, camera_id, name_list)):
            img_path = osp.join(self.img_dir, _img_path[1:]) ## _img_path[1:] -> remove the first '/'
            
            self.pid_container[data_type][pid]=name
            self.cid_container.add(cid)
            self.year_container.add(year)
            classes.add(pid)
            keypoints = None
            sex_id = 0

            meta_info = [pid, name, cid, timestamps[i], is_unique_id[i], _img_path, keypoints, sex_id]
            data.append((img_path, pid, cid, meta_info))

            self.pid_sex_container.add((name, sex_id))

        num_classes = len(classes)

        return data, set(year_list), num_classes
