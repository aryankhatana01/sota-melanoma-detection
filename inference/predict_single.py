import csv
import cv2
from dataset import SIIMISICDataset
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
import geffnet
import os

name = 'hello'

pi1 = 'IP23232'

s1 = 'male'

a1 = 70.0

asgc1 = 'torso'

f1 = '../saved_images/Melanoma.jpg'
img = cv2.imread(f1)
print(img.shape)
with open('../datasettesting.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge", "width", "height", "filepath"])
    writer.writerow([name, pi1, s1, a1, asgc1, img.shape[0], img.shape[1], f1])

df_single_image = pd.read_csv('../datasettesting.csv')

transforms_test = A.Compose([
    A.Resize(640, 640),
    A.Normalize()
])


dataset_test = SIIMISICDataset(df_single_image, 'test', 'test', transform=transforms_test)
image = dataset_test[0]

image = image.to('cpu').unsqueeze(0) 

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(backbone, pretrained=load_pretrained)
        self.dropout = nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x

models = []
for i_fold in range(5):
    model = enetv2('efficientnet_b7', n_meta_features=0, out_dim=9)
    model = model.to('cpu')
    model_file = os.path.join('../best_models', f'9c_b7ns_1e_640_ext_15ep_best_fold{i_fold}.pth')
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    models.append(model)
# print(len(models))

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

with torch.no_grad():
    probs = torch.zeros((image.shape[0], 9)).to('cpu')
    for model in models:
        for I in range(8):
            l = model(get_trans(image, I))
            # print(l)
            probs += l.softmax(1)
probs /= len(models) * 8
# print(probs)

prediction = torch.argmax(probs).item()
# print(prediction)

if prediction == 6:
    print('Melanoma')