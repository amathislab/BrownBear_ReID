from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A

class CustomCoarseDropout(ImageOnlyTransform):
    def __init__(self, max_holes, max_height, max_width, min_holes=1, fill_value=0, p=0.5):
        super(CustomCoarseDropout, self).__init__(p=p)
        self.dropout = A.CoarseDropout(max_holes=max_holes, max_height=max_height, max_width=max_width,
                                       min_holes=min_holes, fill_value=fill_value, always_apply=True)
    
    def apply(self, img, **params):
        return self.dropout(image=img)['image']