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


    # print (len(df.iloc[train_val_indices])/ len(df.iloc[test_indices]))

    if require_id_balance == True:
        train_val_indices, test_indices = balance_animal_id_between_train_val_and_test(df, index2id, id2index, train_val_indices, test_indices)
        
        check_balance(train_val_indices, test_indices, index2id)
        
    train_val_df = df.iloc[train_val_indices]
    test_df = df.iloc[test_indices]             
    return train_val_df, test_df



class DatasetBuilder:
    def __init__(self, data_dir='./data', anno_dir='experiments/heads_6years'):
        self.data_dir = data_dir
        self.anno_dir = anno_dir

        self.FINAL_YEAR = 2022 if '6years' in self.anno_dir else 2021

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
        print(f"Save to {csv_path}")
        return csv_path

    def get_year_ids(self, df_full, year):
        df = df_full[df_full['year'] == year]
        ids = df['id'].unique()
        num_ids = df.groupby(['id']).nunique()['image']
        return ids, num_ids

    def get_unique_ids(self, df_full, year, other_years):
        target_year_ids, _ = self.get_year_ids(df_full, year)
        other_year_ids = []
        for y in other_years:
            ids, _ = self.get_year_ids(df_full, y)
            other_year_ids.extend(ids)
        
        other_year_ids = np.unique(np.array(other_year_ids))
        unique_id = set(target_year_ids) - set(other_year_ids) 
        
        print("other years", other_years)
        print("other years ids", len(other_year_ids))
        print("year:", year, "full ids:", len(target_year_ids), "unique ids:", len(unique_id))
        print("unique ids:", unique_id)
        print("===========================================================================")
        
        return unique_id
    
    def analyze_unique_ids(self, df_full, years):
        unique_ids = {}
        for year in years:
            other_years = list(set(years) - set([year]))
            unique_ids[year] = self.get_unique_ids(df_full, year, other_years)
        return unique_ids
    
    
    def analysis_ids_distribution(self):
        df = self.df_full
        
        num_ids = df.groupby(['id', 'year']).nunique()['image']
        num_ids_full_years = df.groupby(['id']).nunique()['image']
        
        id_list = num_ids_full_years.index
        
        ids_in_years = {}
        ids_in_years_list = []
        for id in id_list:
            ids_data = {}
            data = []
            for year in range(2017, self.FINAL_YEAR+1):
                try:
                    num = num_ids[(id, year)]
                except:
                    num = 0
                data.append(num)
                
            #print("data", id, data)
            ids_in_years[id]=data
            ids_in_years_list.append(data)
            
        ids_in_years_list = np.array(ids_in_years_list).T
        
        #print("ids_in_years_list", ids_in_years_list)
        
        if self.FINAL_YEAR == 2021:
            df = pd.DataFrame(columns=['id', '2017', '2018', '2019', '2020', '2021'])
            df['id'] = id_list
            df['2017'] = ids_in_years_list[0]
            df['2018'] = ids_in_years_list[1]
            df['2019'] = ids_in_years_list[2]
            df['2020'] = ids_in_years_list[3]
            df['2021'] = ids_in_years_list[4]
        else:
            df = pd.DataFrame(columns=['id', '2017', '2018', '2019', '2020', '2021', '2022'])  
            df['id'] = id_list
            df['2017'] = ids_in_years_list[0]
            df['2018'] = ids_in_years_list[1]
            df['2019'] = ids_in_years_list[2]
            df['2020'] = ids_in_years_list[3]
            df['2021'] = ids_in_years_list[4]
            df['2022'] = ids_in_years_list[5]
        
        df.to_csv(f"{ROOT_FOLDER}/ids.csv")



    def split_dataset(self, df, test_year=2017):
        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'].notna()]
        # OOD Test Set
        ood_test_set = df[df['year'] == test_year]
        # Data for Train and IID Test Set
        train_iid_data = df[df['year'] != test_year]
        # Initialize DataFrame to hold train and iid test sets
        train_set = pd.DataFrame(columns=df.columns)
        iid_test_set = pd.DataFrame(columns=df.columns)

        # Group data by 'id' and split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        unique_ids = train_iid_data['id'].unique()
        
        ## NOTE: THIS VERSION IS FOR RANDOMLY PICKING FROM EACH ID
        for individual_id in unique_ids:
            individual_data = train_iid_data[train_iid_data['id'] == individual_id]
            
            # If all images for an individual are from the same day, add to train set
            if individual_data['timestamp'].dt.date.nunique() == 1:
                train_set = pd.concat([train_set, individual_data])
            else:
                # Dynamically determine test_size to maintain the ratio between 80/20 to 90/10
                total_dates = individual_data['timestamp'].dt.date.nunique()
                test_size = max(1, min(0.2, 0.1 / (total_dates / individual_data.shape[0])))
                
                # Split data ensuring no date overlap between train and iid test sets
                for train_idx, test_idx in gss.split(individual_data, groups=individual_data['timestamp'].dt.date):
                    train_subset = individual_data.iloc[train_idx]
                    test_subset = individual_data.iloc[test_idx]
                    # Add to train and iid test sets
                    train_set = pd.concat([train_set, train_subset])
                    iid_test_set = pd.concat([iid_test_set, test_subset])

        ### check intersection
        iid_test_set, train_set = self.check_test_train_intersection(iid_test_set, train_set)
        ood_test_set, train_set = self.check_test_train_intersection(ood_test_set, train_set)

        ## extract 10% of the train_set randomly
        val_set = train_set.sample(frac=0.1, random_state=42)
        ## remove the val_set from the train_set
        train_set = train_set.drop(val_set.index)
        
        return train_set, iid_test_set, ood_test_set, val_set
    
    def split_one_year_data(self, df, ratio_thres=5, split_by_encounter=False, refine=False, outlier_dir=None, year=2017):
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
        
        ## pick some IDs which has more unique dates
        count_id_dates = df.groupby('id')['date'].nunique()
        # get the ids which has more than 2 unique dates
        unique_ids_more_dates = count_id_dates[count_id_dates > 5].index
        # ranomly shuffle the unique_ids and pick 90%
        unique_ids_more_dates = np.random.choice(unique_ids_more_dates, int(len(unique_ids_more_dates) * 0.9), replace=False)
        
        for individual_id in unique_ids:
            
            individual_data = df[df['id'] == individual_id]

            if individual_id not in unique_ids_more_dates:
                ## if the individual_id has less than 5 unique dates (sampled), add to train_val
                train_val = individual_data
                test = pd.DataFrame(columns=df.columns)
            else:
                
                # unique_date = individual_data['timestamp'].dt.date.unique()
                # train_val, test = self.split_data_with_date_constraints(individual_data)
                train_val, test = split_train_val_test(individual_data, 
                                                    require_id_balance = False, 
                                                    ratio_thres=ratio_thres,
                                                    split_by_encounter=split_by_encounter)  ### From the original splitting code 
            
                ## check unique date in train and test
                if len(test['timestamp'].dt.date.unique()) < 2:
                    print(f"test: {individual_id} has less than 2 unique dates")
                
            train_val_set = pd.concat([train_val_set, train_val])
            test_set = pd.concat([test_set, test])

        ### check intersection of items
        test_set, train_val_set = self.check_test_train_intersection(test_set, train_val_set)

        # Extract 10% of the train_val_set randomly for validation
        val_set = train_val_set.sample(frac=0.1, random_state=42)
        # Remove the val_set from the train_val_set for training
        print("len", len(df), len(train_val_set), len(val_set), len(test_set))
        train_set = train_val_set.drop(val_set.index)

        ## check intersection of dates
        inter = self._check_date_or_encounter_intersection(train_set, test_set, split_by_encounter=split_by_encounter)
        if len(inter) > 0:
            raise ValueError("There is an date overlap of same ID between train and IID set")
        else:
            return train_set, val_set, test_set
    


    def split_data_with_date_constraints(self, individual_data):
        """
            Splits the data for an individual ID with more than 4 unique dates, ensuring at least 2 unique dates are in the test set.
        """
        unique_dates = individual_data['timestamp'].dt.date.unique()

        # Handle different scenarios based on the number of unique dates
        if len(unique_dates) != 1:
            # Randomly choose 2 unique dates for the test set
            test_size = max(2, int(len(unique_dates) * 0.3))
            test_dates = np.random.choice(unique_dates, size=test_size, replace=False)
            ### if the test_dates are invalid, remove them from the test_dates
            test_dates = [d for d in test_dates if str(d)[:4]!="0000"]

        elif len(unique_dates) == 1:
            # When there is only 1 unique date, assign all records to the train_val set
            test_dates = []


        test_subset = individual_data[individual_data['timestamp'].dt.date.isin(test_dates)]
        train_val_subset = individual_data[~individual_data['timestamp'].dt.date.isin(test_dates)]

        return train_val_subset, test_subset

    def extract_5years_data_from_6years(self, train_set, iid_test_set, ood_test_set, year=2017):
        '''
        Extract 5 years data from 6 years data
        
        Return:
            new_train_set: get the data except $year and 2022 from train_set
            new_iid_test_set: get the data except $year and 2022  from iid_test_set
            new_ood_test_set: get the all data from $year from ood_test_set
        '''
        
        new_train_set = train_set[(train_set['year'] != year) & (train_set['year'] != 2022)]
        new_iid_test_set = iid_test_set[(iid_test_set['year'] != year) & (iid_test_set['year'] != 2022)]
        new_ood_test_set = ood_test_set[ood_test_set['year'] == year]

        return new_train_set, new_iid_test_set, new_ood_test_set


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
        # df = df.sort_values('timestamp')
        df = df.sort_values(by=['timestamp', 'raw_image'])
    
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
            if time_diff <= time_threshold or image_sequence_diff in range(-5, 6):
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




    def update_splits_with_new_full_data(self, full_csv=None, split_csv=None):
        '''
        Update the splits with the new full data
        '''
        df_full = self.get_dataframe(full_csv)
        df_split = self.get_dataframe(split_csv)

        df_split['image_adjusted'] = df_split['image'].str[5:]

        # Merge df_split with df_full based on the 'image' column
        # This operation will keep all rows from df_split that have a corresponding match in df_full
        # and will append the additional columns from df_full to df_split_new
        df_split_new = pd.merge(df_split, df_full, left_on='image_adjusted', right_on='image', how='inner')

        # Optionally, you can drop the adjusted image column if it's no longer needed
        df_split_new.drop('image_adjusted', axis=1, inplace=True)
        df_split_new.drop('image_x', axis=1, inplace=True)
        ## change column image_y to image
        df_split_new.rename(columns={'image_y': 'image'}, inplace=True)

        # Reset index, if desired
        df_split_new.reset_index(drop=True, inplace=True)

        ## save the new df_split_new to the same file
        df_split_new.to_csv(split_csv, index=False)
        
        print("df_split_new", len(df_split_new))
        print("df_split", len(df_split))
        return df_split_new


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

    data_dir='./data'
    id_csv = 'experiments/full_data/ids.csv'
    anno_dir='experiments/full_data'
    anno_dir_6years='experiments/heads_6years_day_low2train'
    os.makedirs(anno_dir_6years, exist_ok=True)
    outlier_dir='experiments/heads_6years_day/low_performance'
    
    background_dir = './data/backgrounds'

    dataset_builder = DatasetBuilder(data_dir=data_dir, anno_dir=anno_dir_6years)
    years = range(2017, dataset_builder.FINAL_YEAR+1)

    splits_out_dir = f'{anno_dir_6years}/splits'

    full_csv = os.path.join(anno_dir, 'full_data.csv')
    df_6year = dataset_builder.get_dataframe(full_csv)
    ## check if head_pose_13,head_pose_5 in the columns - if yes, remove
    if 'head_pose_13' in df_6year.columns:
        df_6year.drop('head_pose_13', axis=1, inplace=True)
    if 'head_pose_5' in df_6year.columns:
        df_6year.drop('head_pose_5', axis=1, inplace=True)
    if 'focal_length' in df_6year.columns:
        df_6year.drop('focal_length', axis=1, inplace=True)
    if 'focal_length_raw' in df_6year.columns:
        df_6year.drop('focal_length_raw', axis=1, inplace=True)
    
    ## remove the invalid date
    df_6year['timestamp'] = pd.to_datetime(df_6year['timestamp'], errors='coerce')
    df_6year = df_6year[df_6year['timestamp'].notna()]
    
    df_6year['date'] = pd.to_datetime(df_6year['timestamp']).dt.date
    
    ### define photo events (encounters) for each ID in each year
    
    # df_6year['encounter'] = [0] * len(df_6year)
    # df_6year = df_6year.sort_values(by=['timestamp', 'raw_image'])
    
    # analyze unique id
    df_ids = pd.read_csv(id_csv)
    unique_ids = dataset_builder.analyze_unique_ids(df_6year, years)
    unique_id_list = [ list(unique_ids[year]) for year in years]
    unique_id_list = [item for sublist in unique_id_list for item in sublist]
    print("unique_id_list", len(unique_id_list), unique_id_list)
    
    df_ids['is_unique'] = df_ids['id'].apply(lambda x: 1 if x in unique_id_list else 0)
    df_ids.to_csv(id_csv, index=False)
    
    # df_6year_new = pd.DataFrame()
    
    # for year in years:
    #     bg_csv = os.path.join(background_dir, f'{year}_backgrounds.csv')
    #     df_bg = dataset_builder.get_dataframe(bg_csv)
    #     if year != 2022:
    #         df_bg['image'] = df_bg['image'].apply(lambda x: os.path.join(f'{year}_heads/images/', x))
    #     else:
    #         df_bg['image'] = df_bg['image'].apply(lambda x: os.path.join(f'{year}/{year}_heads/images/', x))
                
    #     df_year = df_6year[df_6year['year'] == year]
    #     print(f"year: {year}, len(df_year): {len(df_year)}")
    #     unique_ids = df_year['id'].unique()
    #     print(f"unique ids: {len(df_year['id'].unique())}")
        
    #     # merge with background by image
    #     df_year = pd.merge(df_year, df_bg, on='image', how='left')
    #     print(len(df_bg), len(df_year), df_year.columns)
        
        # ## encounter for each year
        # encounter_counter = 1
        # for id in unique_ids:
        #     df_id = df_year[df_year['id'] == id]
        #     df_year[df_year['id'] == id], encounter_counter = dataset_builder.identify_encounters(df_id, time_threshold=300, encounter_counter=encounter_counter)
        
        ## TODO: smooth the background by encounter
        ## same encounter, same background
        
        # df_6year[df_6year['year'] == year] = df_year 
        # df_6year_new = pd.concat([df_6year_new, df_year])
    
    # # ## save the full data to the csv
    # dataset_builder.save_to_csv(df_6year_new, filename='full_data_new.csv', out_dir=anno_dir, overwrite=True)


    split_by_encounter = False
    
    years = years
    ## PREPARE THE DATA -- SPLIT INTO TRAIN, VAL, IID_TEST
    for year in years:

        print(f"=========================================== YEAR {year} ===========================================", )

        df_year = df_6year[df_6year['year'] == year]
        # split by date or encounter
        df_year_train, df_year_val, df_year_test_iid = dataset_builder.split_one_year_data(df_year, 
                                                                                           ratio_thres=10, 
                                                                                           split_by_encounter=split_by_encounter, 
                                                                                           refine=True, 
                                                                                           outlier_dir=outlier_dir, 
                                                                                           year=year)

        out_dir = os.path.join(splits_out_dir, f'{year}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        overwrite = True
        train_iid_csv = dataset_builder.save_to_csv(df_year_train, filename=f'train_iid.csv', out_dir=out_dir, overwrite=overwrite)
        val_iid_csv = dataset_builder.save_to_csv(df_year_val, filename=f'val_iid.csv', out_dir=out_dir, overwrite=overwrite)
        test_iid_csv = dataset_builder.save_to_csv(df_year_test_iid, filename=f'test_iid.csv', out_dir=out_dir, overwrite=overwrite)

        
    ### PREPARE THE DATA -- SPLIT INTO TRAIN, IID_TEST, OOD_TEST
    for test_year in years:
        df_OOD = df_6year[df_6year['year'] == test_year]

        df_train = pd.DataFrame(columns=df_6year.columns)
        df_iid_test = pd.DataFrame(columns=df_6year.columns)
        df_val = pd.DataFrame(columns=df_6year.columns)

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

        exp_out_dir = os.path.join(anno_dir_6years, f'test_on_{test_year}')
        if not os.path.exists(exp_out_dir):
            os.makedirs(exp_out_dir)

        overwrite = True
        train_iid_csv = dataset_builder.save_to_csv(df_train, filename=f'train_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        val_iid_csv = dataset_builder.save_to_csv(df_val, filename=f'val_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        test_iid_csv = dataset_builder.save_to_csv(df_iid_test, filename=f'test_iid.csv', out_dir=exp_out_dir, overwrite=overwrite)
        ood_test_csv = dataset_builder.save_to_csv(df_OOD, filename=f'test_ood.csv', out_dir=exp_out_dir, overwrite=overwrite)
        


    print("=========================================== START ANALYZING DATA ===========================================")
    df_info = pd.DataFrame(columns=['test_year', 'total_iid', 'train_ids', 'n_train', 'n_val', 'n_test_iid', 'n_test_ood'])
    
    for test_year in years:

        df_train = dataset_builder.get_dataframe(os.path.join(anno_dir_6years, f'test_on_{test_year}', 'train_iid.csv'))
        df_val = dataset_builder.get_dataframe(os.path.join(anno_dir_6years, f'test_on_{test_year}', 'val_iid.csv'))
        df_test_iid = dataset_builder.get_dataframe(os.path.join(anno_dir_6years, f'test_on_{test_year}', 'test_iid.csv'))
        df_OOD_test = dataset_builder.get_dataframe(os.path.join(anno_dir_6years, f'test_on_{test_year}', 'test_ood.csv'))
        
        df_info = df_info.append({'test_year': test_year,
                                    'train_ids': len(df_train['id'].unique()),
                                    'n_train': len(df_train),
                                    'n_val': len(df_val),
                                    'n_test_iid': len(df_test_iid),
                                    'n_test_ood': len(df_OOD_test),
                                    'total_iid': len(df_train) + len(df_val) + len(df_test_iid),
                                    }, ignore_index=True,
                                 )

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
        
    df_info.to_csv(os.path.join(anno_dir, 'data_info.csv'), index=False)
    print("save to ", os.path.join(anno_dir, 'data_info.csv'))
