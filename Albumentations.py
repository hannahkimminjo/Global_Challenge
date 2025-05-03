import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
    ToTensorV2()
])
