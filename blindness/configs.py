import os

from .utils import ON_KAGGLE

data_path = '../input/aptos2019-blindness-detection'

dataset_map = {
    "train": os.path.join(data_path, 'train.csv'),
    "test": os.path.join(data_path, 'test.csv'),
    "fold": 'folds.csv',
    "train_images": os.path.join(data_path, 'train_images'),
    "test_images": os.path.join(data_path, 'test_images'),
    "submission": os.path.join(data_path, 'sample_submission.csv')
}

diabetic_retinopathy_path = '../input/diabetic-retinopathy-resized'

diabetic_retinopathy_map = {
    "train": os.path.join(diabetic_retinopathy_path, 'trainLabels_cropped.csv'),
    "train_images": os.path.join(diabetic_retinopathy_path, 'resized_train_cropped'),
}

if ON_KAGGLE:
    diabetic_retinopathy_map['train_images'] = \
        os.path.join(diabetic_retinopathy_path, 'resized_train_cropped', 'resized_train_cropped')