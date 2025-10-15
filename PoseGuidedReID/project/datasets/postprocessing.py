import numpy as np
import torch
import os
import random
import logging
import pandas as pd
from project.processor import run_evaluation_pipeline
from project.utils.data_analysis import plot_cmc_curve
from project.utils.tools import get_feature_result_path, save_results2csv, get_prediction_result_path
import wandb
import cv2

class PostProcessing():

    def __init__(self, 
                 notes=None,
                 logger=None,
                 ):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger("project.postprocessing")
            self.logger.info('start postprocessing the model predictions...')

    def extract_saved_feature(self, feature_path, split_type='full'):
        """
        Args:
            feature_path: str, path to the saved feature
            split_type: str, 'unique', 'full' or ''
        Returns:
            paths: list, paths of the images
            labels: numpy array, labels of the images
            features: numpy array, features of the images
            cameras: list, cameras of the images
            timestamps: list, timestamps of the images
            is_unique: list, is_unique of the images
        """

        result = np.load(feature_path, allow_pickle=True)
        self.logger.info(f'load the data from:{feature_path}')
        features = torch.FloatTensor(result['feature'])
        paths = result['path']
        labels = result['label']
        cameras = result['camera']
        timestamps = result['date']
        is_unique = result['unique_ids']
        ids = result['id']

        if split_type == 'full':
            self.logger.info('use the full set...')
            return paths, labels, features, cameras, timestamps, is_unique, ids
        else:
            if split_type =='unique': ### only keep the unique bears (novel bears)
                self.logger.info('only keep the unique bears (novel bears)')
                indices = np.where(is_unique == 1)[0]
            elif split_type=='':  ### keep the non-unique bears (known bears)
                self.logger.info('keep the non-unique bears (known bears)')
                indices = np.where(is_unique != 1)[0]
            paths = paths[indices]
            labels = labels[indices]
            features = features[indices]
            cameras = cameras[indices]
            timestamps = timestamps[indices]
            is_unique = is_unique[indices]
            ids = ids[indices]

            return paths, labels, features, cameras, timestamps, is_unique, ids


    def merge_gallery_set_by_results(self, gallery_results_list):

        for i, gallery_results in enumerate(gallery_results_list):
            g_paths, g_labels, g_feature, g_cameras, g_timestamps, g_unique_ids, g_ids = gallery_results
            if isinstance(g_feature, list):
                g_feature = torch.stack(g_feature)
            if i==0:
                gallery_paths = g_paths
                gallery_labels = g_labels
                gallery_feature = g_feature
                gallery_cameras = g_cameras
                gallery_timestamps = g_timestamps
                gallery_unique_ids = g_unique_ids
                gallery_ids = g_ids
            else:
                gallery_paths = np.concatenate((gallery_paths, g_paths))
                gallery_labels = np.concatenate((gallery_labels, g_labels))
                gallery_feature = torch.cat((gallery_feature, g_feature), axis=0)
                gallery_cameras = np.concatenate((gallery_cameras, g_cameras))
                gallery_timestamps = np.concatenate((gallery_timestamps, g_timestamps))
                gallery_unique_ids = np.concatenate((gallery_unique_ids, g_unique_ids))
                gallery_ids = np.concatenate((gallery_ids, g_ids))
        return gallery_paths, gallery_labels, gallery_feature, gallery_cameras, gallery_timestamps, gallery_unique_ids, gallery_ids


    def get_query_gallery_set_from_paths(self, query_feature_path, gallery_feature_path, split_type):
        """
        Args:
            query_feature_path: str, path to the saved query feature
            gallery_feature_path: str, path to the saved gallery feature    
            split_type: str, 'unique', 'full' or ''
        Returns:
            query_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            gallery_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]

        Notes: 
            usually - query_feature_path is the output feature of test_iid or test_ood set
                    - gallery_feature_path is the output feature of train set
        """

        self.logger.info(f'get query-gallery set from paths...')

        query_results = self.extract_saved_feature(query_feature_path, split_type)
        gallery_results = self.extract_saved_feature(gallery_feature_path, split_type='full')  ### always use full for gallery set
        return query_results, gallery_results

    def generate_query_gallery_set_with_additional_gallery(self, 
                                                            pred_feature_path, 
                                                            gallery_feature_path, 
                                                            seed=None, 
                                                            output_dir=None,
                                                            split_type='full',
                                                            query_type='iid',
                                                            gallery_type='splitted'):
        """
        Args:
            pred_feature_path: str, path to the saved feature
            gallery_feature_path: str, path to the saved gallery feature
            seed: int, random seed
            output_dir: str, directory to save the splits
            split_type: str, 'unique', 'full' or ''
        Returns:
            query_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            merged_gallery_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            split_output_path: str, path to save the splits
        Notes: 
            usually - pred_feature_path is the output feature of test_iid or test_ood set
                        - it will be further split into query and gallery set (G1)
                    - gallery_feature_path is the output feature of train set (G2)
                    - final gallery set will be G1 + G2
        """
        
        self.logger.info('generate query-gallery set with additional gallery data...')
        
        # Generate query and initial gallery set by splitting pred_feature_path data
        query_results, initial_gallery_results, split_output_path = self.generate_query_gallery_set(pred_feature_path, 
                                                                                 seed=seed, 
                                                                                 output_dir=output_dir,
                                                                                 split_type=split_type,
                                                                                query_type=query_type,
                                                                                gallery_type=gallery_type)

        # Merge additional gallery data
        additional_gallery_results = self.extract_saved_feature(gallery_feature_path, split_type='full')

        # Combine initial and additional gallery sets
        merged_gallery_results = self.merge_gallery_set_by_results([initial_gallery_results, additional_gallery_results])
        
        return query_results, merged_gallery_results, split_output_path


    def get_query_gallery_set_from_csvs(self, pred_feature_path, output_dir=None, query_csv=None, gallery_csv=None, split_type='full'):
        
        """
        Args:

            pred_feature_path: str, path to the saved feature
            output_path: str, directory to save the splits
            query_csv: str, path to the query csv file
            gallery_csv: str, path to the gallery csv file
            split_type: str, 'unique', 'full' or ''
        Returns:
            query_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            gallery_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]

        Notes:
            we extract the filenames from the existing query, gallery csv files and then get the query-gallery pairs
        
        """

        output_dir = os.path.join(output_dir, 'result_splits')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.basename(pred_feature_path).split('.')[0] + f'_splits_from_csv.npz'
        output_path = os.path.join(output_dir, filename)

        ### check if pred_feature_path is a list of paths
        if isinstance(pred_feature_path, list):
            results = []
            for path in pred_feature_path:
                res_feature = self.extract_saved_feature(path)
                results.append(res_feature)
            results = self.merge_gallery_set_by_results(results)
        else:
            results = self.extract_saved_feature(pred_feature_path)
                
        splits = self.get_query_gallery_pairs(results, query_csv, gallery_csv, output_path=output_path)
        split_key = '' if split_type == '' else f'{split_type}_'
        query_results = splits[f'{split_key}query']
        gallery_results = splits[f'{split_key}gallery']
        query_results = np.array(query_results).T.tolist()
        gallery_results = np.array(gallery_results).T.tolist()

        return query_results, gallery_results, output_path
    

    def generate_query_gallery_set(self, pred_feature_path, output_dir=None, seed=None, split_type='full', 
                                                                                        query_type='iid',
                                                                                        gallery_type='train',
                                                                                        split_single_year=None,):
        """
        Args:
            pred_feature_path: str, path to the saved feature
            output_path: str, directory to save the splits
            seed: int, random seed, if None, we get the splits from the file
            split_type: str, 'unique', 'full' or '' or 'random'
        Returns:
            query_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            gallery_results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            output_path: str, path to save the splits
        Notes:
            we split the data into query and gallery sets and save the splits -- using the random seed
        """
        
        output_dir = os.path.join(output_dir, 'result_splits', query_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.basename(pred_feature_path).split('.')[0] + f'Q{query_type}_G{gallery_type}_splits_s{seed}.npz'
        output_path = os.path.join(output_dir, filename)

        results = self.extract_saved_feature(pred_feature_path)
        
        if split_type != 'random':
            splits = self.split_data_to_query_gallery(results, output_path=output_path, seed=seed, split_single_year=split_single_year)
        else:
            splits = self.split_data_to_query_gallery_random(results, output_path=output_path, seed=seed, split_single_year=split_single_year)

        split_key = '' if split_type == '' else f'{split_type}_'
        query_results = splits[f'{split_key}query']
        gallery_results = splits[f'{split_key}gallery']
        query_results = np.array(query_results, dtype=object).T.tolist()
        gallery_results = np.array(gallery_results, dtype=object).T.tolist()

        return query_results, gallery_results, output_path


    def get_query_gallery_pairs(self, results, query_csv, gallery_csv):
        """
        Args:
            results: list, [paths, labels, features, cameras, timestamps, is_unique, ids]
            query_csv: str, path to the query csv file
            gallery_csv: str, path to the gallery csv file
        Returns:
            res_splits: dict, keys: query, gallery, unique_query, unique_gallery, full_query, full_gallery

        Notes:
            we extract the filenames from the existing query, gallery csv files and then get the query-gallery pairs
        
        """
        ## read the csv files
        
        query_df = pd.read_csv(query_csv)
        gallery_df = pd.read_csv(gallery_csv)

        query_paths = query_df['image'].tolist()
        query_labels = query_df['id'].tolist()

        gallery_paths = gallery_df['image'].tolist()
        gallery_labels = gallery_df['id'].tolist()

        paths, labels, features, cameras, timestamps, is_unique, ids = results

        query_indices = self.get_indices(query_paths, paths)
        gallery_indices = self.get_indices(gallery_paths, paths)

        query_features = features[query_indices]
        gallery_features = features[gallery_indices]


        query_cameras = cameras[query_indices]
        gallery_cameras = cameras[gallery_indices]

        query_timestamps = timestamps[query_indices]
        gallery_timestamps = timestamps[gallery_indices]

        query_is_unique = is_unique[query_indices]
        gallery_is_unique = is_unique[gallery_indices]

        query_ids = ids[query_indices]
        gallery_ids = ids[gallery_indices]

        query_list = []
        gallery_list = []
        for i, (path, label, feature, camera, timestamp, is_unique, id_) in enumerate(zip(query_paths, query_labels, query_features, query_cameras, query_timestamps, query_is_unique, query_ids)):
            query_list.append([path, label, feature, camera, timestamp, is_unique, id_])

        for i, (path, label, feature, camera, timestamp, is_unique, id_) in enumerate(zip(gallery_paths, gallery_labels, gallery_features, gallery_cameras, gallery_timestamps, gallery_is_unique, gallery_ids)):
            gallery_list.append([path, label, feature, camera, timestamp, is_unique, id_])

        res_splits = {
            'query': query_list,
            'gallery': gallery_list,
            'unique_query': [],
            'unique_gallery': [],
            'full_query': query_list,
            'full_gallery': gallery_list,
        }

        return res_splits



    def get_indices(self,
                    target_paths,
                    full_paths):
        """
        Args:
            target_paths: list, paths of the images in the split
            full_paths: list, paths of the images in the dataset
        Returns:
            indices: numpy array, indices of the image paths in the dataset
        """
        
        indices = []
        # print(f'get indices for {full_paths} images...', len(full_paths), len(target_paths))
        for im_path in target_paths:
            # print(f'get indices for {im_path}')
            if im_path not in full_paths:
                raise ValueError(f'Invalid image path: {im_path}')
            idx = np.where(full_paths == im_path)[0][0]
            indices.append(idx)
        return np.array(indices)


    def get_set_info(self,
                     results_data,
                     splits,
                     split_type='query',):
        
        """
        Args:
            splits: dict, keys: query, gallery, unique_query, unique_gallery, full_query, full_gallery
            split_type: str, query, gallery, unique_query, unique_gallery, full_query, full_gallery
        Returns:
            labels: list, labels of the images in the split
            features: numpy array, features of the images in the split
            paths: list, paths of the images in the split
            cameras: list, cameras of the images in the split
            timestamps: list, timestamps of the images in the split
            is_unique: list, is_unique of the images in the split
        """
        try:
            split_data = splits[split_type]
        except:
            raise ValueError(f'Invalid split_type: {split_type}, we only support query, gallery, unique_query, unique_gallery, full_query, full_gallery')
        
        paths, labels, features, cameras, timestamps, is_unique, ids = results_data

        s_paths = [d[0] for d in split_data]
        s_labels = np.array([d[1] for d in split_data])

        indices = self.get_indices(paths, s_paths)

        try:
            s_features = torch.FloatTensor(features[indices])
            s_cameras = cameras[indices]
            s_timestamps = timestamps[indices]
            s_is_unique = is_unique[indices]
            s_ids = ids[indices]
            return s_paths, s_labels, s_features, s_cameras, s_timestamps, s_is_unique, s_ids
        except:
            raise ValueError(f'Empty! No data in {split_type}.')
            

        
    def split_data_to_query_gallery(self,
                                    results,
                                    output_path,
                                    seed=10,
                                    split_single_year=None
                                    ):
        """
        Args:
            output_path: str, path to save the splits
            seed: int, random seed
        Returns:
            res_splits: dict, keys: query, gallery, unique_query, unique_gallery, full_query, full_gallery
        """
        random.seed(seed)

        # self.paths, self.labels, self.features, self.cameras, self.timestamps, self.is_unique, self.ids = results
        paths, labels, features, cameras, timestamps, is_unique, ids = results

        # get date from timestamp 
        dates = []
        for timestamp in timestamps:
            d = timestamp.split(' ')[0]
            dates.append(d)

        # Combine the info into a list of dictionaries
        image_info_list = [
            {"path": path.rstrip(), 
             "label": label, 
             "camera": camera, 
             "date": date, 
             "unique_id": unique_id, 
             "feature": feature,
             "id": id}
            for path, label, camera, date, unique_id, feature, id in zip(paths, labels, cameras, dates, is_unique, features, ids)
        ]

        # Group images by label
        images_by_label = {}
        for data in image_info_list:
            if data["label"] not in images_by_label:
                images_by_label[data["label"]] = []
            images_by_label[data["label"]].append(data)
            
        query_list = []
        gallery_list = []
        unique_query_list = []
        unique_gallery_list = []
        full_query_list = []
        full_gallery_list = []

        # Randomly pick 50% dates from each label for the query set
        for label, images in images_by_label.items():
            random.shuffle(images)

            # Separate images by date
            images_by_date = {}
            for image in images:
                if image["date"] not in images_by_date:
                    images_by_date[image["date"]] = []
                images_by_date[image["date"]].append(image)

            # Sort dates by the number of images
            sorted_dates = sorted(images_by_date.keys(), key=lambda x: len(images_by_date[x]), reverse=True)

            if len(sorted_dates) == 1: ## if the bear images are taken from 1 day, ignored
                continue

            # Pick some non-overlapped date id(s) for the query set (50% of the dates)
            num_dates_for_query = 1 if len(sorted_dates) == 2 else int(len(sorted_dates)/2)
            query_dates = random.sample(sorted_dates, num_dates_for_query)
            gallery_dates = list(set(sorted_dates).difference(query_dates))

            ### check if there is intersection between query and gallery dates
            if len(set(query_dates).intersection(gallery_dates)) > 0:
                raise ValueError('Invalid split, there is intersection between query and gallery dates!!')

            # Add images to the query set and gallery set
            for image in images:

                ### if split_single_year is True, we only split the data from a single year
                if split_single_year is not None:
                    if str(split_single_year) != str(image['date'][:4]):
                        continue

                # data_ = [image["path"], image['label']]
                camera = 0 # image["camera"] ### we don't use camera info
                data_ = [image["path"], image['label'], image['feature'], camera, image['date'], image['unique_id'], image['id']]
                                
                if image["date"] in query_dates:
                    ### QUERY - SET
                    if image["unique_id"] == 1:  ### if the bear is unique in this year
                        unique_query_list.append(data_)
                    else:
                        query_list.append(data_)
                    
                    full_query_list.append(data_)

                else:
                    ### GALLERY - SET
                    if image["unique_id"] == 1:  ### if the bear is unique in this year
                        unique_gallery_list.append(data_)
                    else:
                        gallery_list.append(data_)
                    full_gallery_list.append(data_)

        res_splits = {
            'query': query_list,
            'gallery': gallery_list,
            'unique_query': unique_query_list,
            'unique_gallery': unique_gallery_list,
            'full_query': full_query_list,
            'full_gallery': full_gallery_list,
        }

        res_splits_image_label_only = {
            'query': np.array(query_list, dtype=object)[:,:2],
            'gallery': np.array(gallery_list, dtype=object)[:,:2],
            'full_query': np.array(full_query_list, dtype=object)[:,:2],
            'full_gallery': np.array(full_gallery_list, dtype=object)[:,:2],
        }

        if len(unique_query_list) > 0:
            res_splits_image_label_only['unique_query'] = np.array(unique_query_list, dtype=object)[:,:2]
        if len(unique_gallery_list) > 0:
            res_splits_image_label_only['unique_gallery'] = np.array(unique_gallery_list, dtype=object)[:,:2]
        
        np.savez(output_path, **res_splits_image_label_only)
        self.logger.info(f'splitted query-gallery saved to : {output_path}')

        return res_splits


    def split_data_to_query_gallery_random(self,
                                        results,
                                        output_path,
                                        seed=10,
                                        split_single_year=None
                                        ):
        """
        Args:
            output_path: str, path to save the splits
            seed: int, random seed
        Returns:
            res_splits: dict, keys: query, gallery, unique_query, unique_gallery, full_query, full_gallery
        """
        random.seed(seed)

        # self.paths, self.labels, self.features, self.cameras, self.timestamps, self.is_unique, self.ids = results
        paths, labels, features, cameras, timestamps, is_unique, ids = results

        # get date from timestamp 
        dates = []
        for timestamp in timestamps:
            d = ''
            if timestamp is not None:
                d = timestamp.split(' ')[0]
            dates.append(d)

        # Combine the info into a list of dictionaries
        image_info_list = [
            {"path": path.rstrip(), 
             "label": label, 
             "camera": camera, 
             "date": date, 
             "unique_id": unique_id, 
             "feature": feature,
             "id": id}
            for path, label, camera, date, unique_id, feature, id in zip(paths, labels, cameras, dates, is_unique, features, ids)
        ]

        # Group images by label
        images_by_label = {}
        for data in image_info_list:
            if data["label"] not in images_by_label:
                images_by_label[data["label"]] = []
            images_by_label[data["label"]].append(data)
            
        query_list = []
        gallery_list = []
        unique_query_list = []
        unique_gallery_list = []
        full_query_list = []
        full_gallery_list = []

        # Randomly pick 50% images from each label for the query set
        for label, images in images_by_label.items():
            random.shuffle(images)

            for image in images:
                ### if split_single_year is True, we only split the data from a single year
                if split_single_year is not None:
                    if str(split_single_year) != str(image['date'][:4]):
                        continue
                    
            ### randomly pick 50% image index for query set
            num_images_for_query = int(len(images)/2)
            ## get the random index
            query_images_indices = random.sample(range(len(images)), num_images_for_query)
            query_images_indices = np.array(query_images_indices)
            gallery_images_indices = set(range(len(images))).difference(query_images_indices)
            gallery_images_indices = np.array(list(gallery_images_indices))

            query_images = [[images[ind]["path"], images[ind]['label'], images[ind]['feature'], 0, images[ind]['date'], images[ind]['unique_id'], images[ind]['id']] for ind in query_images_indices]
            gallery_images = [[images[ind]["path"], images[ind]['label'], images[ind]['feature'], 0, images[ind]['date'], images[ind]['unique_id'], images[ind]['id']] for ind in gallery_images_indices]
            
            query_list.extend(query_images)
            gallery_list.extend(gallery_images)
            full_query_list.extend(query_images)
            full_gallery_list.extend(gallery_images)

        res_splits = {
            'random_query': query_list,
            'random_gallery': gallery_list,
            'random_unique_query': [],
            'random_unique_gallery': [],
            'random_full_query': full_query_list,
            'random_full_gallery': full_gallery_list,
        }

        res_splits_image_label_only = {
            'query': np.array(query_list, dtype=object)[:,:2],
            'gallery': np.array(gallery_list, dtype=object)[:,:2],
            'full_query': np.array(full_query_list, dtype=object)[:,:2],
            'full_gallery': np.array(full_gallery_list, dtype=object)[:,:2],
        }

        if len(unique_query_list) > 0:
            res_splits_image_label_only['unique_query'] = np.array(unique_query_list, dtype=object)[:,:2]
        if len(unique_gallery_list) > 0:
            res_splits_image_label_only['unique_gallery'] = np.array(unique_gallery_list, dtype=object)[:,:2]
        
        output_path = output_path.replace('.npz', '_random.npz')
        np.savez(output_path, **res_splits_image_label_only)
        self.logger.info(f'splitted query-gallery saved to : {output_path}')

        return res_splits


    def eval_query_gallery(self,
                           cfg,
                            split_type='full',
                            query_type='iid',
                            gallery_type='splitted',
                            output_dir=None,
                            seed=10,
                            top_k=10,
                            wb_run=None):
        """

        Args:
        query_type: 'iid' or 'ood', or 'val'
        gallery_type: 'train' or 'train_w_ood' or 'split'

        
        """
        if query_type == 'val':
            if cfg.DATASETS.NAMES != 'lion':
                data_type = 'val_iid'
            else:
                data_type = 'val'
        else:
            data_type = f'test_{query_type}'

        weight = cfg.TEST.WEIGHT  ## net_xxx.pth
        epoch = weight.split('_')[-1].split('.')[0]

        if gallery_type == 'train':
            ### query is from part of test set, gallery is from train set
            query_pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type=data_type, epoch=epoch)
            gallery_pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type='train_iid', epoch=epoch)

            query_data, gallery_data = self.get_query_gallery_set_from_paths(query_feature_path=query_pred_feature_path, 
                                                                                        gallery_feature_path=gallery_pred_feature_path,
                                                                                        split_type=split_type)
            split_output_path = ""
            
        elif gallery_type == 'train_w_ood':
            ### query is from part of test set, gallery is from train_iid + another part of test set

            pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type=data_type, epoch=epoch)
            gallery_pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type='train_iid', epoch=epoch)

            query_data, gallery_data, split_output_path = self.generate_query_gallery_set_with_additional_gallery(pred_feature_path=pred_feature_path,
                                                                                                                    gallery_feature_path=gallery_pred_feature_path,
                                                                                                                    seed=seed, 
                                                                                                                    output_dir=output_dir,
                                                                                                                    split_type=split_type)
        elif gallery_type == 'split':
            ### query and gallery are random splitted from test set
            pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type=data_type, epoch=epoch)
            print("get feature from : pred_feature_path", pred_feature_path)
            query_data, gallery_data, split_output_path = self.generate_query_gallery_set(pred_feature_path, 
                                                                                        seed=seed, 
                                                                                        output_dir=output_dir,
                                                                                        split_type=split_type)

        
        final_results_outdir = os.path.join(output_dir, 'final_results', query_type)  ### save the final results
        if not os.path.exists(final_results_outdir):
            os.makedirs(final_results_outdir)
        cmc, mAP, all_AP, sorted_qg_indices, n_query, n_gallery, results, result_image = run_evaluation_pipeline(cfg,
                                                                                                                query_data,
                                                                                                                gallery_data,
                                                                                                                query_type=query_type,
                                                                                                                gallery_type=gallery_type,
                                                                                                                split_type=split_type,
                                                                                                                top_k=top_k)


        ### 
        ## bgr to rgb
        res_file_name = f'results_Q{query_type}_G{gallery_type}_s{seed}'
        predictions = wandb.Image(
                    result_image[:, :, ::-1], caption=f"{res_file_name}"
                    )
        wandb.log({'{}'.format(res_file_name): predictions})
        
        output_path = os.path.join(final_results_outdir, f'{res_file_name}.csv')
        image_output_path = os.path.join(final_results_outdir, f'{res_file_name}.png')
        cv2.imwrite(image_output_path, result_image)
        print('prediction image save to: ', image_output_path)
        save_results2csv(results, output_path)


        ### save the results
        if split_type == '':
            split_type = 'known'
        elif split_type == 'unique':
            split_type = 'novel'
        elif split_type == 'random':
            split_type = 'random'
        else:
            split_type = 'full'
        split_type = f'{query_type}-{split_type}'
        
        
        
        result_dict = {'seed': seed, 
                    'epoch': epoch,
                    'gallery': gallery_type, 
                    'split_type': split_type, 
                    'n_query':n_query, 
                    'n_gallery':n_gallery, 
                    'mAP': mAP, 
                    'splits_path': split_output_path, 
                    'result_path': output_path,
                    'result_images': image_output_path
                    }
        if cfg.TEST.MAX_RANK >= 10:
            # results_dict
            # append ranks to result_dict
            for i in range(10):
                result_dict[f'r{i+1}'] = cmc[i]
            
        
        ## plot cmc curve
        title=f'Query-{split_type}_Gallery-{gallery_type}_seed{seed}'
        cmc_results_dir = os.path.join(final_results_outdir, 'cmc_curves')
        if not os.path.exists(cmc_results_dir):
            os.makedirs(cmc_results_dir)
        output_path = os.path.join(cmc_results_dir, f'cmc_{title}.jpg')
        plot_cmc_curve(cmc, title=title, output_dir=output_path, wb_run=wb_run)
        ### save cmc to csv
        cmc_csv = os.path.join(final_results_outdir, 'cmc_curves', f'cmc_Query-{split_type}_Gallery-{gallery_type}.csv')
        ### save cmc to a row 
        cmc_df = pd.DataFrame(cmc.reshape(1, -1), columns=[f'r{i+1}' for i in range(len(cmc))])
        ### put seed into first column
        cmc_df.insert(0, 'seed', seed)
        ## if the file exists, append the new row
        if os.path.exists(cmc_csv):
            cmc_df.to_csv(cmc_csv, mode='a', header=False, index=False)
        else:
            cmc_df.to_csv(cmc_csv, index=False)
        result_dict['cmc_curve'] = cmc_csv

        return result_dict


    def eval_testset(self,
                        cfg,
                        query_type='',
                        output_dir=None,
                        top_k=10,
                        filter_date=True,
                        split_single_year=None, 
                        epoch=None,):
        """
        Args:
            query_type: 'iid' or 'ood', or 'val'

        """

        if query_type == 'val':
            if cfg.DATASETS.NAMES != 'lion':
                data_type = 'val_iid'
            else:
                data_type = 'val'
        else:
            data_type = f'test_{query_type}'

        if epoch is None:
            weight = cfg.TEST.WEIGHT  ## net_xxx.pth
            epoch = weight.split('_')[-1].split('.')[0]

        final_results_outdir = os.path.join(output_dir, 'final_results', 'full')  ### save the final results
        if not os.path.exists(final_results_outdir):
            os.makedirs(final_results_outdir)
        
        pred_feature_path = get_feature_result_path(cfg, out_dir=output_dir, data_type=data_type, epoch=epoch)
        
        results = self.extract_saved_feature(pred_feature_path)
        
        if split_single_year is not None:
            paths, labels, features, cameras, timestamps, is_unique, ids = results
            ## get the index of the timestamps where the year is the same
            indices = [i for i, t in enumerate(timestamps) if int(t[:4]) == int(split_single_year)]
            query_data = [paths[indices], labels[indices], features[indices], cameras[indices], timestamps[indices], is_unique[indices], ids[indices]]
            query_data = np.array(query_data, dtype=object).T  ## only keep the query from the single year
            gallery_data = np.array(results, dtype=object).T  ## keep the gallery as full
            # gallery_data = query_data
        
        else:
            query_data = np.array(results, dtype=object).T
            gallery_data = query_data
            
        cmc, mAP, all_AP, sorted_qg_indices, n_query, n_gallery, results, result_image = run_evaluation_pipeline(cfg,
                                                                                                                    query_data,
                                                                                                                    gallery_data,
                                                                                                                    query_type=query_type,
                                                                                                                    gallery_type='all',
                                                                                                                    split_type='all',
                                                                                                                    top_k=top_k, 
                                                                                                                    filter_date=filter_date)

        csv_path = get_prediction_result_path(final_results_outdir, 
                                   query_type, 
                                   gallery_type='all', 
                                   seed=None, 
                                   split_single_year=None)
        
        image_output_path = csv_path.replace('.csv', '.png')
        cv2.imwrite(image_output_path, result_image)
        print('prediction image save to: ', image_output_path)
        save_results2csv(results, csv_path)

        split_type = f'{query_type}-all'
        mAP_for_max_rank = cfg.TEST.MAX_RANK if cfg.TEST.MAP_MAX_RANK else False
        result_dict = {'epoch': epoch,
                        'gallery': 'all', 
                        'split_type': split_type, 
                        'n_query':n_query, 
                        'n_gallery':n_gallery, 
                        'filter_date': cfg.TEST.FILTER_DATE,
                        'mAP_for_max_rank': mAP_for_max_rank,
                        'split_single_year': split_single_year,
                        'mAP': mAP, 
                    }
        if cfg.TEST.MAX_RANK >= 10:
            for i in range(10):
                result_dict[f'r{i+1}'] = cmc[i]
        if cfg.TEST.MAX_RANK == 20:
            result_dict[f'r20'] = cmc[19]
        result_dict['result_images'] = image_output_path
        result_dict['result_path'] = csv_path
        
        # ## plot cmc curve
        # title=f'{query_type}_results'
        # cmc_results_dir = os.path.join(final_results_outdir, 'cmc_curves')
        # os.makedirs(cmc_results_dir, exist_ok=True)
        # output_path = os.path.join(cmc_results_dir, f'cmc_{title}.jpg')
        # plot_cmc_curve(cmc, title=title, output_dir=output_path, wb_run=wb_run)
        # ### save cmc to csv
        # cmc_csv = os.path.join(final_results_outdir, 'cmc_curves', f'cmc_{title}.csv')
        # ### save cmc to a row 
        # cmc_df = pd.DataFrame(cmc.reshape(1, -1), columns=[f'r{i+1}' for i in range(len(cmc))])
        # ## if the file exists, append the new row
        # if os.path.exists(cmc_csv):
        #     cmc_df.to_csv(cmc_csv, mode='a', header=False, index=False)
        # else:
        #     cmc_df.to_csv(cmc_csv, index=False)
        # result_dict['cmc_curve'] = cmc_csv

        return result_dict


        