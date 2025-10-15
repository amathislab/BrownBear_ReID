import os
import random
import numpy as np
import pandas as pd
import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

np.random.seed(1)
ROOT_FOLDER='experiments/heads_6years/'


def try_frac(time_keys, frac, time2index):
    train_N = int(len(time_keys) * frac)
    train_keys = time_keys[:train_N]
    test_keys = time_keys[train_N:]

    train_sum = 0
    test_sum = 0

    train_indices = []
    test_indices = []
    
    for k in time_keys:
        if k in train_keys:
            train_sum+=len(time2index[k])

            train_indices +=(time2index[k])
        else:
            test_sum+=len(time2index[k])
            test_indices +=time2index[k]
    return (train_sum/test_sum), train_indices, test_indices
    

def build_id_dict(df):
    id2index = {}
    index2id = {}
    for i in range(len(df.index)):
        animal_id = df.iloc[i]['id']
        if animal_id not in id2index:
            id2index[animal_id] = []

        index2id[i] = animal_id
        
        id2index[animal_id].append(i)


    return id2index, index2id

def check_balance(train_val_indices, test_indices, index2id):
    id_in_train_val = set()
    id_in_test = set()
    for index in train_val_indices:
        id_in_train_val.add(index2id[index])
    for index in test_indices:
        id_in_test.add(index2id[index])    

    if (id_in_train_val!=id_in_test):
        print ('train_val', len(id_in_train_val))
        print ('test', len(id_in_test))
    else:
        print ('balanced')
        
def balance_animal_id_between_train_val_and_test(df, index2id, id2index, train_val_indices, test_indices):

    print ('before balancing')
    check_balance(train_val_indices, test_indices, index2id)        
    id_in_train_val = set()
    id_in_test = set()
    for index in train_val_indices:
        id_in_train_val.add(index2id[index])
    for index in test_indices:
        id_in_test.add(index2id[index])


    diff = id_in_train_val - id_in_test   
    for id in diff:
        # remove the first one
        index = id2index[id][0]
        assert len(id2index[id]) > 1, f"only {len(id2index[id])} image in that id {id}"

        test_indices.append(index)
        train_val_indices.remove(index)


    diff = id_in_test - id_in_train_val
    for id in diff:

        # remove the first one
        index = id2index[id][0]
        assert len(id2index[id]) > 1, f"only {len(id2index[id])} image in that id {id}"
        test_indices.remove(index)
        train_val_indices.append(index)        


    print ('after balancing')
    check_balance(train_val_indices, test_indices, index2id)    

            
    return train_val_indices, test_indices

def split_train_val_test(df, require_id_balance = False, ratio_thres=1.5, split_by_encounter=False):
    temp = []
    time2index = {}

    id2index, index2id = build_id_dict(df)
    
    for i in range(len(df.index)):
        if not split_by_encounter:
            timestamp = df.iloc[i]['timestamp']
            date = str(timestamp)[:10]
            if date not in time2index:
                time2index[date]  = []
            time2index[date].append(i)
        else:
            encounter = df.iloc[i]['encounter']
            if encounter not in time2index:
                time2index[encounter]  = []
            time2index[encounter].append(i)
            
    time_keys = np.array(list(time2index.keys()))
    shuffle = np.random.permutation(len(time_keys))
    time_keys = time_keys[shuffle]

    work = False
    min_dist = 1000
    min_frac = None
    for frac in np.arange(0.1, 0.8, 0.01):
        ## try to find the ratio that is close to the ratio_thres
        ratio, train_val_indices, test_indices = try_frac(time_keys, frac, time2index)
        if np.abs(ratio - ratio_thres) < 0.1:
            work = True
            break
        if np.abs(ratio - ratio_thres) < min_dist:
            min_dist = np.abs(ratio-ratio_thres)
            min_frac = frac                
    if not work:
        ratio, train_val_indices, test_indices = try_frac(time_keys, min_frac, time2index)        


    print (len(df.iloc[train_val_indices])/ len(df.iloc[test_indices]))

    if require_id_balance == True:
        train_val_indices, test_indices = balance_animal_id_between_train_val_and_test(df, index2id, id2index, train_val_indices, test_indices)
        
        check_balance(train_val_indices, test_indices, index2id)
        
    train_val_df = df.iloc[train_val_indices]
    test_df = df.iloc[test_indices]             
    return train_val_df, test_df



class DatasetBuilder:
    def __init__(self, data_dir='./datasets_other_animals/MacaqueFaces'):
        self.data_dir = data_dir

    def get_dataframe(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except:
            raise ValueError(f"File {csv_path} does not exist")
    
    
    def save_to_csv(self, df, filename, out_dir, overwrite=False):

        csv_path = os.path.join(out_dir, filename)
        if os.path.exists(csv_path) and not overwrite:
            raise ValueError(f"File {csv_path} exists, if you want to overwrite, set overwrite=True")
        df.to_csv(csv_path, index=False)
        return csv_path


    def split_one_year_data(self, df, ratio_thres=5, split_by_encounter=False):
        """
            Split one year data into train, val, test, for the same year
                # 1. train -  the data for training
                # 2. validation - data for validation
                    (in-domain data vs train -- the bear images *may* from the same day for same ID - just randomly splitted from the training data), 
                # 3. test - the data for IID test
                    (the bear images for same ID are from the same year, but different days from the train) -> will be used for temporoal IID for different experiments
        """
        # Convert 'timestamp' to datetime and remove invalid date
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'].notna()]

        # Initialize DataFrame to hold train_val and test sets
        train_val_set = pd.DataFrame(columns=df.columns)
        test_set = pd.DataFrame(columns=df.columns)
        
        unique_ids = df['id'].unique()
        
        ## NOTE: THIS VERSION IS FOR RANDOMLY PICKING FROM EACH ID
        for individual_id in unique_ids:
            individual_data = df[df['id'] == individual_id]

            train_val, test = split_train_val_test(individual_data, 
                                                    require_id_balance = False, 
                                                    ratio_thres=ratio_thres,
                                                    split_by_encounter=split_by_encounter)  ### From the original splitting code 
            
            train_val_set = pd.concat([train_val_set, train_val])
            test_set = pd.concat([test_set, test])

        ### check intersection of items
        test_set, train_val_set = self.check_test_train_intersection(test_set, train_val_set)

        # Extract 10% of the train_val_set randomly for validation
        val_set = train_val_set.sample(frac=0.1, random_state=42)
        # Remove the val_set from the train_val_set for training
        train_set = train_val_set.drop(val_set.index)

        ## check intersection of dates
        inter = self._check_date_or_encounter_intersection(train_set, test_set, split_by_encounter=split_by_encounter)
        if len(inter) > 0:
            raise ValueError("There is an date overlap of same ID between train and IID set")
        else:
            return train_set, val_set, test_set
    
    def identify_encounters(self, df, time_threshold=120, encounter_counter=1):
        """
        Identifies encounters based on the timestamp and raw_image name.

        Args:
            df (pd.DataFrame): DataFrame with 'timestamp' and 'raw_image' columns.
            time_threshold (int): Number of seconds to consider for an encounter.

        Returns:
            pd.DataFrame: DataFrame with an additional 'encounter' column.
        """

        # Sort the dataframe by timestamp just in case it's not sorted
        df = df.sort_values('timestamp')
        
        # Convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Initialize encounter list
        encounters = [encounter_counter]

        # Helper function to extract the numeric part of the image name
        def _extract_number(image_name):
            numbers = ''.join(filter(str.isdigit, image_name))
            return int(numbers)
    
        # Iterate through the DataFrame and identify encounters
        for i in range(1, len(df)):
            # Calculate the time difference in seconds
            time_diff = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
            
            prev_name = os.path.basename(df.iloc[i-1]['raw_image'])
            curr_name = os.path.basename(df.iloc[i]['raw_image'])

            # Extract numeric part of image names
            prev_num = _extract_number(prev_name)
            curr_num = _extract_number(curr_name)
            # Check the sequence in image naming and time threshold
            image_sequence_diff = curr_num - prev_num

            # If the time difference is less than the threshold and the image sequence diff is 1 or 2, it's the same encounter
            if time_diff <= time_threshold and image_sequence_diff in [-2, -1, 0 , 1, 2]:
                encounters.append(encounter_counter)
            else:
                encounter_counter += 1
                encounters.append(encounter_counter)

        # Add the encounter column to the DataFrame
        df['encounter'] = encounters
        return df, max(encounters)

    def check_test_train_intersection(self, df_test, df_train):
        '''
        Check if there is an intersection between test and train, if yes, remove the intersection from test
        '''
        df_test_images = set(df_test['image'])
        df_train_images = set(df_train['image'])
        intersection = df_test_images.intersection(df_train_images)
        ### if there is an intersection, print the intersection
        if len(intersection)>0:
            print("before: df_test", len(df_test))
            print("intersection", len(intersection))
            df_test = df_test[~df_test['image'].isin(intersection)]
            print("after: df_test", len(df_test))
        else:
            print("No intersection in the items")
        
        return df_test, df_train




    def _check_date_or_encounter_intersection(self, df_train, df_test, split_by_encounter=False):
        '''
            Check the date intersection between train and test of the same identity
        '''
        
        if split_by_encounter:
            split_type = 'encounter'
        else:
            split_type = 'date'
            df_train['date'] = pd.to_datetime(df_train['timestamp']).dt.date
            df_test['date'] = pd.to_datetime(df_test['timestamp']).dt.date

        ### group by id and date
        train_grouped = df_train.groupby(['id', split_type, 'year']).size().reset_index(name='count')
        test_grouped = df_test.groupby(['id', split_type, 'year']).size().reset_index(name='count')
        ### get the intersection
        intersection = pd.merge(train_grouped, test_grouped, on=['id', split_type, 'year'], how='inner')
        ### unique ids in the intersection
        unique_ids = intersection['id'].unique()
        ### merge df_train and df_test
        df_train_test = pd.concat([df_train, df_test])
        ### get the grouped id and data in the merged df_train_test
        train_test_grouped = df_train_test.groupby(['id', split_type, 'year']).size().reset_index(name='count')
        
        return intersection
    
    



if __name__ == "__main__":

    data_dir='datasets_other_animals/MacaqueFaces'
    
    anno_dir_2years=os.path.join(data_dir, 'anno_2years')
    
    dataset_builder = DatasetBuilder(data_dir=data_dir)
    full_csv = os.path.join(data_dir, 'MacaqueFaces_ImageInfo.csv')
    
    df_2year = dataset_builder.get_dataframe(full_csv)
    df_2year['timestamp'] = pd.to_datetime(df_2year['DateTaken'])
    df_2year['year'] = pd.DatetimeIndex(df_2year['DateTaken']).year
    years = df_2year['year'].unique()
    
    ## for each image, get the path, image = os.path.join(path, image_name)
    df_2year['image'] = df_2year.apply(lambda x: os.path.join(x['Path'], x['FileName']), axis=1)
    # count the number of images in each year
    
    splits_out_dir = f'{anno_dir_2years}/splits'
    if not os.path.exists(splits_out_dir):
        os.makedirs(splits_out_dir)
    
    ## save the full data to the csv
    full_csv = os.path.join(anno_dir_2years, 'full_data.csv')
    dataset_builder.save_to_csv(df_2year, filename='full_data.csv', out_dir=anno_dir_2years, overwrite=True)
    
    
    indices = np.arange(len(df_2year))
    # get 80% random indices for train, 20% for test
    train_indices = np.random.choice(indices, int(0.8*len(df_2year)), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    
    train_set = df_2year.iloc[train_indices]
    test_set = df_2year.iloc[test_indices]
    
    random_out_dir = os.path.join(data_dir, 'random')
    train_csv = dataset_builder.save_to_csv(train_set, filename='train.csv', out_dir=random_out_dir, overwrite=False)
    test_csv = dataset_builder.save_to_csv(test_set, filename='test.csv', out_dir=random_out_dir, overwrite=False)
    
    
    
    
'''
    split_by_encounter = False
    ### PREPARE THE DATA -- SPLIT INTO TRAIN, VAL, IID_TEST
    for year in years:

        print(f"=========================================== YEAR {year} ===========================================", )

        df_year = df_2year[df_2year['year'] == year]
        # split by date or encounter
        df_year_train, df_year_val, df_year_test_iid = dataset_builder.split_one_year_data(df_year, ratio_thres=10, split_by_encounter=split_by_encounter)

        out_dir = os.path.join(splits_out_dir, f'{year}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        overwrite = True
        train_iid_csv = dataset_builder.save_to_csv(df_year_train, filename=f'train_iid.csv', out_dir=out_dir, overwrite=overwrite)
        val_iid_csv = dataset_builder.save_to_csv(df_year_val, filename=f'val_iid.csv', out_dir=out_dir, overwrite=overwrite)
        test_iid_csv = dataset_builder.save_to_csv(df_year_test_iid, filename=f'test_iid.csv', out_dir=out_dir, overwrite=overwrite)

        
    ### PREPARE THE DATA -- SPLIT INTO TRAIN, IID_TEST, OOD_TEST
    for test_year in years:
        df_OOD = df_2year[df_2year['year'] == test_year]

        df_train = pd.DataFrame(columns=df_2year.columns)
        df_iid_test = pd.DataFrame(columns=df_2year.columns)
        df_val = pd.DataFrame(columns=df_2year.columns)

        ## extract data from other 5 years
        for year in years:
            if year == test_year:
                continue            
            
            train_csv = os.path.join(splits_out_dir, f'{year}', 'train_iid.csv')
            train_set = dataset_builder.get_dataframe(train_csv)

            test_iid_csv = os.path.join(splits_out_dir, f'{year}', 'test_iid.csv')
            test_iid_set = dataset_builder.get_dataframe(test_iid_csv)

            val_iid_csv = os.path.join(splits_out_dir, f'{year}', 'val_iid.csv')
            val_set = dataset_builder.get_dataframe(val_iid_csv)

            ### print
            print(f"year: {year}, train_set: {len(train_set)}, test_iid_set: {len(test_iid_set)}, val_set: {len(val_set)}")

            df_train = pd.concat([df_train, train_set])
            df_iid_test = pd.concat([df_iid_test, test_iid_set])
            df_val = pd.concat([df_val, val_set])

        ## check intersection of items
        df_iid_test, df_train = dataset_builder.check_test_train_intersection(df_iid_test, df_train)
        df_OOD, df_train = dataset_builder.check_test_train_intersection(df_OOD, df_train)
        df_val, df_train = dataset_builder.check_test_train_intersection(df_val, df_train)

        exp_out_dir = os.path.join(anno_dir_2years, f'test_on_{test_year}')
        if not os.path.exists(exp_out_dir):
            os.makedirs(exp_out_dir)

        overwrite = True
        train_iid_csv = dataset_builder.save_to_csv(df_train, filename=f'train_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        val_iid_csv = dataset_builder.save_to_csv(df_val, filename=f'val_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        test_iid_csv = dataset_builder.save_to_csv(df_iid_test, filename=f'test_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        ood_test_csv = dataset_builder.save_to_csv(df_OOD, filename=f'test_ood.csv', out_dir=exp_out_dir, overwrite=overwrite)
        


    print("=========================================== START ANALYZING DATA ===========================================")
    for test_year in years:

        df_train = dataset_builder.get_dataframe(os.path.join(anno_dir_2years, f'test_on_{test_year}', 'train_iid.csv'))
        df_val = dataset_builder.get_dataframe(os.path.join(anno_dir_2years, f'test_on_{test_year}', 'val_iid.csv'))
        df_test_iid = dataset_builder.get_dataframe(os.path.join(anno_dir_2years, f'test_on_{test_year}', 'test_iid.csv'))
        df_OOD_test = dataset_builder.get_dataframe(os.path.join(anno_dir_2years, f'test_on_{test_year}', 'test_ood.csv'))

        print("len(df_2year)", len(df_2year))
        print("len(df_OOD_test)", len(df_OOD_test))
        print("len(df_train)", len(df_train) ) #//len(df_2year[df_2year['year'] != test_year])*100)
        print("len(df_val)", len(df_val)) #//len(df_2year[df_2year['year'] != test_year])*100)
        print("len(df_test_iid)", len(df_test_iid)) #//len(df_2year[df_2year['year'] != test_year])*100)
        print("len(df_train) + len(df_val) + len(df_test_iid) + len(df_OOD_test)", len(df_train) + len(df_val) + len(df_test_iid) + len(df_OOD_test))

        total_iid = len(df_train) + len(df_val) + len(df_test_iid)
        ### roughly between 6:1:3 for train : val : test_iid
        print("ratio", len(df_train)/total_iid * 100, len(df_val)/total_iid* 100, len(df_test_iid)/total_iid* 100)

        ### check if df_train contains data from OOD
        df_train_ood_year = df_train[df_train['year'] == test_year]
        if len(df_train_ood_year) > 0:
            raise ValueError("There is an OOD year in the train set")

        ### 
        split_type = 'encounter' if split_by_encounter else 'date'
        # check the date intersection between train and test_iid
        inter_dates = dataset_builder._check_date_or_encounter_intersection(df_train, df_test_iid, split_by_encounter=split_by_encounter)
        if len(inter_dates) > 0:
            raise ValueError(f"There is {split_type} overlap of same ID between train and IID set in {test_year}")
        # check the date intersection between train and test_ood
        inter_dates = dataset_builder._check_date_or_encounter_intersection(df_train, df_OOD_test, split_by_encounter=split_by_encounter)
        if len(inter_dates) > 0:
            raise ValueError(f"There is {split_type} overlap of same ID between train and OOD set in {test_year}")
'''