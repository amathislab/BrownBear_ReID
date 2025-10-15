import os
import rawpy
import pandas as pd
from PIL import Image
import json
import glob
import argparse
from tqdm import tqdm

def get_raw_csv(raw_csv_path):
    raw_df = pd.read_csv(raw_csv_path)
    return raw_df

def read_image(image_path):
    if image_path.endswith('.NEF'):
        with rawpy.imread(image_path) as raw:
            image = raw.postprocess()
            image = Image.fromarray(image)
    else:
        image = Image.open(image_path)
    return image

def crop_body_images(item, bear_id, raw_data, output_dir):
    
    image_path = item['file']
    n_detections = len(item['detections'])
    
    if n_detections > 1:
        output_dir = os.path.join(output_dir, bear_id, 'multiple')
    else:
        output_dir = os.path.join(output_dir, bear_id, 'single')
    
    os.makedirs(output_dir, exist_ok=True)
    
    body_data = {
        'image': [],
        'id': [],
        'timestamp': [],
        'camera': [],
        'raw_image': [],
        'body_bbox': []
    }
    
    for i, detection in enumerate(item['detections']):
        box = detection['bbox']
        img = read_image(image_path)
        w, h = img.size
        w_margin = w*0.03
        h_margin = h*0.03
        x1 = box[0] * w 
        y1 = box[1] * h 
        x2 = (box[0] + box[2]) * w 
        y2 = (box[1] + box[3]) * h 
        
        new_bbox = [
            x1-w_margin if x1-w_margin>=0 else x1,
            y1-h_margin if y1-h_margin>=0 else y1,
            x2+w_margin if x2+w_margin<=w else x2,
            y2+h_margin if y2+h_margin<=h else y2,
        ]
        img_cropped = img.crop(new_bbox)
        
        new_img_name = os.path.basename(image_path).split(".")[0] + "_" + str(i) + f'_{bear_id}' + ".JPG"
        new_img_path = os.path.join(output_dir, new_img_name)
        img_cropped.save(new_img_path)
        
        body_data['image'].append(new_img_path)
        body_data['id'].append(bear_id)
        body_data['timestamp'].append(raw_data['timestamp'].values[0])
        body_data['camera'].append(raw_data['camera'].values[0])
        body_data['raw_image'].append(raw_data['image'].values[0])
        body_data['body_bbox'].append(new_bbox)
        
    return body_data


def main():
    
    ### set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_detection_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--raw_csv_path', type=str, default='')
    args = parser.parse_args()
    
    ### 
    body_detection_path = args.body_detection_path
    output_dir = args.output_dir
    raw_csv_path = args.raw_csv_path
    body_csv_path = raw_csv_path.replace('raw.csv', 'body.csv')
    
    df_raw = get_raw_csv(raw_csv_path)
    
    ### check if body_detection_path is a directory or json file
    if os.path.isdir(body_detection_path):
        ### get all json files 
        json_files = glob.glob(os.path.join(body_detection_path, '*.json'))
    else:
        json_files = [body_detection_path]
    
    body_items = {
        'image': [],
        'id': [],
        'timestamp': [],
        'camera': [],
        'raw_image': [],
        'body_bbox': []
    }
    
    for json_file in tqdm(json_files):
        ## load json file
        with open(json_file, 'r') as f:
            detection_results = json.load(f)
        
        bear_id = os.path.basename(json_file).replace('.json', '').replace('output_', '')
        
        for item in detection_results["images"]:
            filepath = item['file']
            ### get raw image item
            raw_data = df_raw[df_raw['image'] == filepath]
            
            body_data = crop_body_images(item, bear_id, raw_data, output_dir)
            for key in body_data.keys():
                body_items[key].extend(body_data[key])
    
    body_df = pd.DataFrame(body_items)
    
    body_df.to_csv(body_csv_path, index=False)
    print('body_df', body_csv_path)
    
if __name__ == "__main__":
    main()