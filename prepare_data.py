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

model = torch.hub.load(repo_or_dir="facebookresearch/dinov3",
    model="dinov3_vitl16", pretrained=False)
model.load_state_dict(torch.load('dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
model.to(device)

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
    upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for data in tqdm.tqdm(train_dataloader):
                skeleton = data['skeleton'].to(device)
                color = data['color'].to(device)
                warped = data['warped'].to(device)
                names = data['name']

                feats = model.get_intermediate_layers(skeleton, n=range(9), reshape=True, norm=True)
                output_skeleton = upsample(feats[-1])
                feats = model.get_intermediate_layers(color, n=range(9), reshape=True, norm=True)
                output_color = upsample(feats[-1])
                output_warped = vae.encode(warped).latent_dist.sample().mul_(0.18215)

                for i in range(output_skeleton.shape[0]):
                    torch.save(output_skeleton[i:i+1], 'data_binary/pose/{}.pt'.format(names[i][:-4]))
                    torch.save(output_color[i:i+1], 'data_binary/color/{}.pt'.format(names[i][:-4]))
                    torch.save(output_warped[i:i+1], 'data_binary/extracted/{}.pt'.format(names[i][:-4]))