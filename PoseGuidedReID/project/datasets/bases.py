import logging
from typing import Tuple
from PIL import Image, ImageFile
from networkx import is_path

from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from pathlib import Path
import json
from typing import List, Optional, Any, Set
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {'image', 'id'}
OPTIONAL_COLUMNS = {
    'camera_id': 0,
    'year': -1,
    'is_unique': 0,
    'timestamp': -1,
    'sex': -1
}
        
        
def get_bear_image_path(img_dir, image_path, is_body_image=False):
    """Construct full image path."""
    ### if start with '20' -> mcneil data
    if image_path[:2] == '20':
        year = image_path[:4]
        if year == '2022' or is_body_image:
            img_path = osp.join(img_dir, image_path)
        else:
            img_path = osp.join(img_dir, str(year), image_path)
    else:
        ### Katmai data (full path saved in csv)
        img_path = image_path
    
    return img_path
       
        
def merge_csv_files(file_list):
    columns = ['id', 'image']
    df = pd.DataFrame(columns=columns)
    for file in file_list:
        df = pd.concat([df, pd.read_csv(file)[columns]], axis=1)
    return df
    
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class BaseDataset(Dataset):
    """
    Base class of reid dataset
    """
    def __init__(self, 
                 root='', 
                 img_dir=None,
                 cfg=None,
                 **kwargs):
        super(BaseDataset, self).__init__()
        
        self.root = osp.abspath(osp.expanduser(root))
        self.cfg = cfg
        self.img_dir = img_dir
        
        self.pid_container = {'train':{}, 'iid':{}, 'ood':{}, 'val':{}}
        self.cid_container = set()
        self.year_container = set()
                
    
    def _check_before_run(self):
        """Check if folder is available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def get_imagedata_info(self, data: List[Tuple]) -> Tuple[int, int, int]:
        pids, cams = [], []
        for img_path, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        return len(pids), len(cams), len(data)
        
    def load_data(self, dir_path):
        """Load data from directory, CSV file, or list of files"""
        path = Path(dir_path)
        
        if path.is_dir():
            # TODO: implement folder processing -> read all subfolders (id) and images from each subfolder -> into DataFrame
            raise NotImplementedError("Folder processing not implemented")
            
        if isinstance(dir_path, list):
            if not all(f.endswith('.csv') for f in dir_path):
                raise ValueError("All files must be CSV format")
            return merge_csv_files(dir_path)
            
        if path.suffix == '.csv':
            return pd.read_csv(path)
            
        raise ValueError(f"Invalid data source {dir_path}")

    def _validate_columns(self, data):
        """Validate required columns exist in dataset."""
        missing_cols = REQUIRED_COLUMNS - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _fill_missing_columns(self, data):
        """Fill missing optional columns with default values."""
        for col, default in OPTIONAL_COLUMNS.items():
            if col not in data.columns:
                logger.info(f"Column '{col}' not found, using default value: {default}")
                data[col] = default
        return data

    def _create_id_mapping(self, data):
        """Create mapping from names to numeric IDs."""
        unique_names = sorted(data['id'].unique())
        return {name: idx for idx, name in enumerate(unique_names)}
    
    def _get_image_path(self, image_path):
        """Construct full image path."""
        if self.cfg is not None:
            if self.cfg.DATASETS.NAMES == 'bear':
                image_path = get_bear_image_path(self.img_dir, image_path)
            elif self.cfg.DATASETS.NAMES == 'base':
                ### check if image_path start with 20
                if not image_path.startswith('20') and osp.exists(image_path):
                    image_path = image_path
                else:
                    ### if image_path start with 20* --> bear
                    image_path = get_bear_image_path(self.img_dir, image_path)
        else:
            image_path = osp.join(self.img_dir, image_path)
        return image_path
    
    def _process_row(self, row, data_type=None):
        """Process a single data row.
        
        Args:
            row: DataFrame row
            data_type: Type of dataset
            
        Returns:
            tuple: (img_path, pid, cid, meta_info)
        """
        # Handle image path

        img_path = self._get_image_path(row['image'])
        
        # Update containers
        pid = row['pid']
        cid = row['camera_id']
        name = row['id']
        
        if data_type is not None:
            self.pid_container[data_type][pid] = name
        
        # Create metadata
        meta_info = [
            pid,
            name,
            cid,
            row['timestamp'],
            row['is_unique'],
            row['image'],
            row['sex']
        ]
                
        return img_path, pid, cid, meta_info

    def process_dir(self, dir_path, data_type=None):
        """Process directory containing dataset CSV file.
        
        Args:
            dir_path: Path to CSV file
            data_type: Type of dataset (train/test/val)
            
        Returns:
            tuple: (processed_data, year_set, num_classes)
            
        Raises:
            ValueError: If required columns are missing
            IOError: If file cannot be read
        """
        logger.info(f'Processing directory: {dir_path}')
        
        # Load and validate data
        try:
            data = self.load_data(dir_path)
            self._validate_columns(data)
        except Exception as e:
            logger.error(f"Error loading data from {dir_path}: {str(e)}")
            raise

        # Fill missing columns with defaults
        data = self._fill_missing_columns(data)
        
        # Create ID mapping
        name_to_pid = self._create_id_mapping(data)
        data['pid'] = data['id'].map(name_to_pid)
                
        # Process each row
        processed_data = []
        classes = set()
        
        self.cid_container.update(set(data['camera_id']))
        self.year_container.update(set(data['year']))
        
        for _, row in data.iterrows():
            try:
                item_data = self._process_row(row, data_type)
                processed_data.append(item_data)
                classes.add(item_data[1])  # pid
            except Exception as e:
                logger.warning(f"Error processing row {row['image']}: {str(e)}")
            
        return processed_data, set(data['year']), len(classes)




class SplittedImageDataset(BaseDataset):
    """
    Base class of reid dataset
    """
    def __init__(self, 
                 root='', 
                 img_dir=None,
                 cfg=None,
                 **kwargs):
        super(SplittedImageDataset, self).__init__()

        self.root = osp.abspath(osp.expanduser(root)) 
        self.img_dir = img_dir
        self.cfg = cfg
        
        self.train_dir = osp.join(self.root, 'train_iid.csv')
        self.test_iid_dir = osp.join(self.root, 'test_iid.csv')
        self.test_ood_dir = osp.join(self.root, 'test_ood.csv')
        self.val_iid_dir = osp.join(self.root, 'val_iid.csv')
        
        self.pid_container = {'train':{}, 'iid':{}, 'ood':{}, 'val':{}}
        
        self.train, self.train_year_list, self.n_train_classes = self.process_dir(self.train_dir, data_type='train')
        self.test_iid, self.test_iid_year_list, self.n_test_iid_classes = self.process_dir(self.test_iid_dir, data_type='iid')
        self.test, self.test_year_list, self.n_test_classes  = self.process_dir(self.test_ood_dir, data_type='ood')
        self.val, self.val_year_list, self.n_val_classes = self.process_dir(self.val_iid_dir, data_type='val')

        self.print_dataset_statistics()


    def print_dataset_statistics(self):

        print("Dataset statistics:")
        print("  --------------------------------------------------------------------")
        print("  subset    | year(min-max) | #ids| #cameras| #images ")
        print("  --------------------------------------------------------------------")

        total_years = f"{min(self.year_container)}-{max(self.year_container)}"
        print(f"  ==== Time period: {total_years} ==== ")
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



class BaseImageDataset(BaseDataset):
    """
    Base class of reid dataset

    Dataset statistics:
    # identities: 
    """

    def __init__(self, 
                 root='', 
                 img_dir=None,
                 cfg=None,
                 data_type='train',
                 **kwargs):
        super(BaseImageDataset, self).__init__()

        self.root = root
        self.img_dir = img_dir
        self.cfg = cfg
        self.data_type = data_type

        self._check_before_run()
        self.data, self.data_year_list, self.n_classes = self.process_dir(self.root, data_type=data_type)

        print("=> Data loaded")
        self.print_dataset_statistics()
        

    def print_dataset_statistics(self):

        print("Dataset statistics:")
        print("  --------------------------------------------------------------------")
        print(f"dataset| #ids: {self.n_classes} | #images: {len(self.data)} ")
        print("  --------------------------------------------------------------------")



class ImageDataset(Dataset):
    
    def __init__(self, 
                 dataset: List[Tuple],
                 transform: Optional[Any] = None,
                 pre_scaling: Optional[int] = None,
                 overfitting: Optional[bool] = False):
        """
        Args:
            dataset: List of tuples (is_path, pid, camid, meta_info)
            transform: Optional transform to apply to images
            pre_scaling: Optional pre-scaling size of the original image for testing, default is None
            overfitting: Optional overfitting flag, default is False
        """
        self.dataset = dataset
        self.transform = transform
        self.pre_scaling = pre_scaling
        self.overfitting = overfitting
        
    def __len__(self) -> int:
        if self.overfitting:
            if len(self.dataset) > 5000:
                return 5000 # for overfitting
            else:
                return len(self.dataset)
        else:
            return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple:
        """Get item by index.
        
        Args:
            index: Index of the item to get
            
        Returns:
            Tuple of (image_tensor, pid, camid, meta_info)
        """
        img_path, pid, camid, meta_info = self.dataset[index]
        img = read_image(img_path)
        
        if self.pre_scaling:
            img = img.resize((self.pre_scaling, self.pre_scaling))
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, pid, camid, meta_info
    
