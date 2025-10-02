import os.path as osp
import torchvision.transforms as transforms
import tqdm
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from diffusers import AutoencoderKL

device = 'cuda'

preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
vae.requires_grad_(False)

class CustomImageDataset(Dataset):
    def __init__(self):
        self.filenames = os.listdir(osp.join('dataset', 'cloth'))

    def __getitem__(self, index):
        file_name = self.filenames[index]

        B_path = osp.join('dataset', 'image', file_name)
        X_path = osp.join('dataset', 'image-parse-v3', file_name.replace('jpg', 'png'))
        person = Image.open(B_path)
        segment = Image.open(X_path).convert('L')
        segment_np = np.array(segment)
        person_np = np.array(person)
        person_np = np.transpose(person_np, (2, 0, 1))
        mask = (segment_np == 126).astype(int)
        mask = np.expand_dims(mask, axis=0)
        extracted_cloth_np = mask * person_np
        extracted_cloth = Image.fromarray(np.transpose(extracted_cloth_np, (1, 2, 0)).astype('uint8'), 'RGB')

        S_path = osp.join('dataset', 'image-densepose', file_name)
        skeleton = Image.open(S_path).convert('RGB')

        C_path = osp.join('dataset', 'cloth', file_name)
        color = Image.open(C_path).convert('RGB')

        return {'warped': preprocess(extracted_cloth), 'skeleton': preprocess(skeleton), 'color': preprocess(color), 'name': file_name}

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    dataset = CustomImageDataset()
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    if not os.path.exists("data_binary/pose"):
        os.makedirs("data_binary/pose")
    if not os.path.exists("data_binary/color"):
        os.makedirs("data_binary/color")
    if not os.path.exists("data_binary/extracted"):
        os.makedirs("data_binary/extracted")
    upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.float32):
            for data in tqdm.tqdm(train_dataloader):
                skeleton = data['skeleton'].to(device)
                color = data['color'].to(device)
                warped = data['warped'].to(device)
                names = data['name']
                output_skeleton = vae.encode(skeleton).latent_dist.sample().mul_(0.18215)
                output_color = vae.encode(color).latent_dist.sample().mul_(0.18215)
                output_warped = vae.encode(warped).latent_dist.sample().mul_(0.18215)
                for i in range(output_skeleton.shape[0]):
                    torch.save(output_skeleton[i], 'data_binary/pose/{}.pt'.format(names[i][:-4]))
                    torch.save(output_color[i], 'data_binary/color/{}.pt'.format(names[i][:-4]))
                    torch.save(output_warped[i], 'data_binary/extracted/{}.pt'.format(names[i][:-4]))