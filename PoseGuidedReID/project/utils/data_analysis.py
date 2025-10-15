
import numpy as np
import pandas as pd
import os
import wandb
import cv2
from .tools import get_full_path
import matplotlib.pyplot as plt
import torch

def save_top_k_matched_images(image_paths, 
                              labels, 
                              timestampes, 
                              target_height=400, 
                              font_scale=0.8, 
                              font_color=(0, 0, 0), 
                              padding=30, 
                              margin=10,
                              dataset='bear',
                              body_input=False):

    """
        image_paths: [query, gallery1, gallery2, ...]
        labels: [query_label, gallery1_label, gallery2_label, ...]
    
    """
    
    

    images = []
    for path in image_paths:
        # if '20' in path[:2]:  ### bear dataset
        full_path = get_full_path(path, img_dir= os.getcwd() + '/../../../data', dataset=dataset, body_input=body_input)
        if not os.path.exists(full_path):
            if '598A4386_BLR_0_Fisher.JPG' in full_path:
                full_path = full_path.replace('598A4386_BLR_0_Fisher.JPG', '598A4386_BLR_0_Sister.jpg')  ## replace the wrong file name
            else:
                try:
                    full_path = get_full_path(path, img_dir= os.getcwd() + '/../../../data',)
                except:
                    print(f'{full_path} does not exist')
                    continue
        img = cv2.imread(full_path)
        resized = cv2.resize(img, (target_height, target_height))
        images.append(resized)
    
    final_images = []
    for i, (img, label) in enumerate(zip(images, labels)):
        if i == 0:
            ### set it to black
            pad_color = (255, 255, 255)
            im_type = 'Q:'
            query_label = label
        else:
            if label == query_label:
                pad_color = (176, 133, 0)
            else:
                pad_color = (32, 50, 220)
                
            im_type = 'G:'
        
        top = bottom = left = right = padding
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        ### pad with white color
        top = bottom = left = right = 1
        padded_img = cv2.copyMakeBorder(padded_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        year = timestampes[i].split(' ')[0] # convert timestampes to datetime
        year = year.replace('-', '') ## remove the dash, from 2020-08-01
        text = f'{im_type} {label}({year})'
        ## put the text on the image
        position = (0, target_height +  padding * 2 - 5)
        padded_img = cv2.putText(padded_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
        final_images.append(padded_img)
    
    concatenated_image = np.hstack(final_images)
    concatenated_image = cv2.copyMakeBorder(concatenated_image, 0, 0, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return concatenated_image
        

def save_batch_results(cfg, query_labels, query_paths, query_timestamps, top_k_g_labels, top_k_g_paths, top_k_g_timestamps, n_sample=5):

    full_images = []
    for i in range(n_sample):
        ### randomly select a query image
        i = np.random.randint(len(query_labels))
        query_label = query_labels[i]
        g_labels = top_k_g_labels[i]
        query_path = query_paths[i]
        query_timestamp = query_timestamps[i]
        gallery_paths = top_k_g_paths[i]
        gallery_timestamps = top_k_g_timestamps[i]

        labels = [query_label] + g_labels
        image_paths = [query_path] + gallery_paths
        timestampes = [query_timestamp] + gallery_timestamps

        result_sample = save_top_k_matched_images(image_paths, labels, timestampes, dataset=cfg.DATASETS.NAMES, body_input=cfg.DATASETS.BODY_INPUT)
        full_images.append(result_sample)

    full_images = np.vstack(full_images)
    return full_images



def get_top_k_matched_images(cfg, 
                             all_ap,
                             query_labels, 
                             query_paths, 
                             query_ids,
                             gallery_labels, 
                             gallery_paths, 
                             gallery_ids,
                             query_timestamps,
                             gallery_timestamps,
                             indices, 
                             k=20,
                             n_sample=10,
                             filter_date=False):
    """
    Args:
        indices: numpy array, shape: (len(query), len(gallery)), sorted indices of gallery images for each query image
        k: int, top k matched images from gallery set
    Returns:
        
    """ 
    # if indices is tensor, convert it to numpy array
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()

    if filter_date:
        num_q = len(query_labels)
        filtered_indices = []
        for q_idx in range(num_q):
            # get query pid and date
            q_label = query_labels[q_idx]
            q_time = query_timestamps[q_idx]
            # remove gallery samples that have the same pid and date with query
            order = indices[q_idx]  # select one row
            remove = (gallery_labels[order] == q_label) & (gallery_timestamps[order] == q_time)
            keep = np.invert(remove)
            filtered_indices.append(order[keep])
        indices = filtered_indices

    top_k_g_paths = [[gallery_paths[i] for i in inds[:k]] for inds in indices]
    top_k_g_labels = [[gallery_labels[i] for i in inds[:k]] for inds in indices]
    top_k_g_ids = [[gallery_ids[i] for i in inds[:k]] for inds in indices]
    top_k_g_timestamps = [[gallery_timestamps[i] for i in inds[:k]] for inds in indices]

    ### save results to csv
    query_labels = np.array(query_labels)
    query_paths = np.array(query_paths)
    query_ids = np.array(query_ids)
    all_ap = np.round(np.array(all_ap), 3)

    # Convert each inner list to a string
    str_top_k_g_paths = ['; '.join(map(str, sublist)) for sublist in top_k_g_paths]
    str_top_k_g_labels = ['; '.join(map(str, sublist)) for sublist in top_k_g_labels]
    str_top_k_g_ids = ['; '.join(map(str, sublist)) for sublist in top_k_g_ids]
    str_top_k_g_timestamps = ['; '.join(map(str, sublist)) for sublist in top_k_g_timestamps]

    ### TODO: save the prediction_samples (half best AP, half worst AP)
    prediction_samples = save_batch_results(cfg, query_labels, query_paths, query_timestamps, top_k_g_labels, top_k_g_paths, top_k_g_timestamps, n_sample=n_sample)  

    ## save results to csv
    results = {'query_labels': query_labels,
                'AP': all_ap,
                'top_k_g_labels': str_top_k_g_labels,
                'query_paths': query_paths,
                'top_k_g_images': str_top_k_g_paths,
                'query_timestamps': query_timestamps,
                'top_k_g_timestamps': str_top_k_g_timestamps,}
    
    return results, prediction_samples




def plot_cmc_curve(cmc, title='', output_dir='', wb_run=None):
    
    plt.plot(cmc, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title(f'CMC Curve-{title}')
    # plt.legend([f'{title}'])
    ### log the plot to wandb
    plt.savefig(output_dir, dpi=300)
    if wb_run is not None:
        wb_run.log({f'cmc_{title}': wandb.Image(output_dir, caption=f'CMC Curve {title}')})
    plt.close()

