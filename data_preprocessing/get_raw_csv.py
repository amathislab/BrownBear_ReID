import numpy as np
import pandas as pd
import os
from PIL import Image, ExifTags
from datetime import datetime
import glob
import argparse
import rawpy
import io
from tqdm import tqdm
import subprocess
import json

def get_metadata(filename):
    command = ["exiftool", "-j", filename]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)
    if 'DateTimeOriginal' in metadata[0]:
        timestamp = datetime.strptime(metadata[0]['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
    elif 'ModifyDate' in metadata[0]:
        timestamp = datetime.strptime(metadata[0]['ModifyDate'], '%Y:%m:%d %H:%M:%S')
    else:
        timestamp = None
    if 'Model' in metadata[0]:
        camera = metadata[0]['Model']
    else:
        camera = None
    if 'ImageWidth' in metadata[0]:
        width = metadata[0]['ImageWidth']
    else:
        width = None
    if 'ImageHeight' in metadata[0]:
        height = metadata[0]['ImageHeight']
    else:
        height = None
    return timestamp, camera, width, height


def process_raw_images(image_dir, img_types=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'nef']):
    
    #### folder structure: image_dir/id/image.jpg
    raw_images = glob.glob(os.path.join(image_dir, '*/*.*'))
    raw_info = {
        'image': [],
        'id': [],
        'timestamp': [],
        'camera': [],
        'width': [],
        'height': [],
    }
    for image in tqdm(raw_images):
        
        ### filter out the image 
        if not image.split('.')[-1].lower() in img_types:
            continue
        
        id = os.path.basename(os.path.dirname(image))
        timestamp, camera, width, height = get_metadata(image)
        if timestamp is None:
            ## put the 0000-00-00 00:00:00 data with datatime
            timestamp = datetime(1970, 1, 1, 0, 0, 0)
        raw_info['image'].append(image)
        raw_info['timestamp'].append(timestamp)
        raw_info['id'].append(id)
        raw_info['camera'].append(camera)
        raw_info['width'].append(width)
        raw_info['height'].append(height)
    
    return raw_info
    
def save_raw_csv(raw_info, save_path):
    df = pd.DataFrame(raw_info)
    df.to_csv(save_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    
    ### process the raw images, save the timestamp and the image name
    
    raw_info = process_raw_images(args.image_dir)
    save_raw_csv(raw_info, args.save_path)
    print('saved to: ', args.save_path)
    

if __name__ == '__main__':
    main()