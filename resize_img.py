import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_img(img_path, save_path, size):
    img = Image.open(img_path)
    img = img.resize(size, Image.BILINEAR)
    img.save(save_path)

input_folder = 'input/'
output_folder = 'output/'

images = glob.glob(os.path.join(input_folder, '*.jpg'))
os.makedirs(output_folder, exist_ok=True)

Parallel(n_jobs=8)(
    delayed(resize_img)(
        img_path,
        os.path.join(output_folder, os.path.basename(img_path)),
        (512, 512)
    ) for img_path in tqdm(images)
)