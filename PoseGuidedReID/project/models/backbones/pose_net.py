import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from .hrnet import HRNet
# from model.poseresnet import PoseResNet
# from models.detectors.YOLOv3 import YOLOv3  # import only when multi-person is enabled
import torch.nn.functional as F
from .pose_utils import transform_preds, get_affine_transform


def _box2cs(box, image_width=256, image_height=256):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width=256, image_height=256):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio = image_width * 1.0 / image_height

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    pixel_std = 200
    scale_thre = 1.25
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_thre

    return center, scale

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2) ## B x 17
    maxvals = np.amax(heatmaps_reshaped, 2) ## B x 17

    maxvals = maxvals.reshape((batch_size, num_joints, 1)) ## B x 17 x 1
    idx = idx.reshape((batch_size, num_joints, 1)) ## B x 17 x 1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) ## B x 17 x 2, like repeat in pytorch

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):

    """
    Get final predictions from heatmaps.
    """

    coords, maxvals = get_max_preds(batch_heatmaps)  ### coords in the heatmap size

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return coords, preds, maxvals


class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 256),  ## because the original resolution we trained for bear is 256x256
                 max_batch_size=32,
                 device="cuda:0"):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): name of the model to use ('HRNet' or 'PoseResNet').
            resolution (tuple): resolution of the input images (height, width).
            max_batch_size (int): maximum batch size to use when predicting on multiple images.

        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.max_batch_size = max_batch_size
        self.device = device

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        print("self.device", self.device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            # print("=> loading checkpoint '{}'".format(checkpoint_path))
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def predict(self, image):
        """
        """
        if len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')


    def _transform_images(self, images):

        center_list = []
        scale_list = []

        ### 
        images_tensor = torch.empty(images.shape[0], 3, self.resolution[0], self.resolution[1])

        for i, image in enumerate(images):
            center, scale = _box2cs([0, 0, image.shape[1], image.shape[0]], self.resolution[1], self.resolution[0])
            trans = get_affine_transform(center, scale, 0, self.resolution)
            image = cv2.warpAffine(
                image,
                trans,
                self.resolution,
                flags=cv2.INTER_LINEAR
            )

            if self.transform:
                image = self.transform(image)

            images_tensor[i] = image
            center_list.append(center)
            scale_list.append(scale)

        return images_tensor, center_list, scale_list
        
    def _predict_batch(self, images):
        # Initialize lists to store results for each subset
        all_heatmaps = []
        all_outputs_feature = []
        all_preds = []
        all_maxvals = []

        ori_image_size = images.shape[2:4]
        
        # Determine the number of images and split size
        num_images = images.shape[0]
        split_size = self.max_batch_size if self.max_batch_size < num_images else num_images

        # Loop through the images in increments of split_size
        for i in range(0, num_images, split_size):
            # Select a subset of images
            images_subset = images[i:i+split_size]

            # Transform images
            images_transformed, center_list, scale_list = self._transform_images(images_subset)

            # Predict with batch
            heatmaps, outputs_feature = self.model(images_transformed, return_feature=True)

            if isinstance(heatmaps, list):
                heatmaps = heatmaps[-1]

            # Prediction on original size
            coords, preds, maxvals = get_final_preds(heatmaps.cpu().detach().numpy(), center_list, scale_list)

            # Store results
            all_heatmaps.append(heatmaps)
            all_outputs_feature.append(outputs_feature)
            all_preds.extend(preds)  # Assuming get_final_preds returns a list
            all_maxvals.extend(maxvals)

        # Concatenate results stored in lists
        all_heatmaps = torch.cat(all_heatmaps, dim=0)
        all_outputs_feature = torch.cat(all_outputs_feature, dim=0)

        # Convert lists to numpy arrays if necessary
        # Note: This step may need adjustments based on the actual return types of get_final_preds
        all_preds = np.array(all_preds)
        all_maxvals = np.array(all_maxvals)

        return all_heatmaps, all_outputs_feature, all_preds, all_maxvals
