import os
import pandas as pd
import glob
import argparse
import shutil


def get_body_data(body_csv_path):
    df = pd.read_csv(body_csv_path)
    return df

def get_meta_data_from_body(df, df_body):
    ### check if 'timestamp', 'camera' in the df_body
    if 'timestamp' in df_body.columns and 'camera' in df_body.columns:
        ### join the df and df_body on the body_image column from left, and image from right , and keep the timestamp, camera column
        ### drop the body_image column
        n_len = len(df)
        images = df['image'].tolist()
        df = df.merge(df_body, left_on='body_image', right_on='image', how='left')
        df = df.drop(columns=['image_y'])
        df = df.drop(columns=['id_y'])
        df = df.rename(columns={'image_x': 'image'})
        df = df.rename(columns={'id_x': 'id'})
        assert len(df) == n_len, f"The length of df is not the same as the original length, {len(df)} vs {n_len}"
        assert set(df['image'].tolist()) == set(images), f"The image column is not the same as the original images"
    return df
    

def is_curated_image_exists(image_path, new_folder='images_curated'):
    """
    Check if the curated image exists
    """
    curated_path = image_path.replace('images_uncurated', new_folder)
    if os.path.exists(curated_path):
        shutil.copy(image_path, curated_path)
        
    return os.path.exists(curated_path), curated_path

def check_curated_head_csv(df_merged, new_folder='images_curated'):
    #### remove the images that are not in the curated folder, and change the image path to the curated folder
    ####
    df_merged['is_exists'], df_merged['new_image'] = zip(*df_merged['image'].apply(is_curated_image_exists, new_folder=new_folder))
    ### remove the images that are not in the curated folder
    df_merged = df_merged[df_merged['is_exists']]
    df_merged['image'] = df_merged['new_image'].copy()
    df_merged.drop(columns=['is_exists', 'new_image'], inplace=True)
    return df_merged

def merge_head_csvs(head_cropped_images_dir, head_csv_path, is_curated, merge_meta_info=False):
    
    ## get all csvs in head_cropped_images_dir
    csv_files = glob.glob(os.path.join(head_cropped_images_dir, '*/*/*.csv'))
    print(head_cropped_images_dir)
    body_csv_path = head_csv_path.replace('head.csv', 'body.csv')

    ## read all csvs -- merge into one dataframe
    ## get column names from first csv
    df = pd.read_csv(csv_files[0])
    column_names = df.columns.tolist()
    df_merged = pd.DataFrame(columns=column_names)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        ### append to df_merged without index and column names
        df_merged = pd.concat([df_merged, df], ignore_index=True, axis=0)
        
    if is_curated:
        #### 
        df_merged = check_curated_head_csv(df_merged)
        head_csv_path = head_csv_path.replace('head.csv', 'head_curated.csv')
        
    ###
    if merge_meta_info:
        try:
            print("body_csv_path======", body_csv_path)
            df_body = get_body_data(body_csv_path)
            df_merged = get_meta_data_from_body(df_merged, df_body)
        except Exception as e:
            print(f"Error merging meta info: {e}")
    
    df_merged.to_csv(head_csv_path, index=False)
    print(f'Head csv merged and saved to {head_csv_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_cropped_images_dir', type=str, required=True)
    parser.add_argument('--head_csv_path', type=str, required=True)
    parser.add_argument('--is_curated', action='store_true')
    parser.add_argument('--merge_meta_info', action='store_true')
    args = parser.parse_args()
    
    merge_head_csvs(args.head_cropped_images_dir, args.head_csv_path, args.is_curated, args.merge_meta_info)


if __name__ == '__main__':
    main()


