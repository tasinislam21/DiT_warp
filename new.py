from dataclasses import dataclass
import numpy as np
import torch.utils.data as data
import torch
import os.path as osp
from MMDiT import MMDiT
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import torchvision
import os
import argparse
dist.init_process_group(backend="nccl")

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, default=32)      # option that takes a value
parser.add_argument('-d', '--depth', type=int, default=20)      # option that takes a value
args = parser.parse_args()

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = args.batch
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def cleanup():
    # Destroy the process group
    dist.destroy_process_group()

def cosine_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = []
    for i in reversed(range(timesteps)):
        T = timesteps - 1
        beta = start + 0.5 * (end - start) * (1 + np.cos((i / T) * np.pi))
        betas.append(beta)
    return torch.Tensor(betas)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_timestep(image, t):
    betas_t = get_index_from_list(betas, t, image.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, image.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, image.shape)
    # Call model (current image - noise prediction)
    with torch.no_grad():
        sample_output = model(image_tokens=image, text_tokens=torch.cat([pose, cloth], 1), time_cond=t)

    model_mean = sqrt_recip_alphas_t * (
            image - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t[0:1].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(image)
        posterior_variance_t = get_index_from_list(posterior_variance, t, image.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(t.device) * x_0.to(t.device) \
    + sqrt_one_minus_alphas_cumprod_t.to(t.device) * noise.to(t.device), noise.to(t.device)


T = 10
betas = cosine_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

class BaseDataset(data.Dataset):
    def __init__(self, padding = 32):
        super(BaseDataset, self).__init__()
        self.name_list = os.listdir('data_binary/extracted')

    def __getitem__(self, index):
        file_name = self.name_list[index]
        B_path = osp.join('data_binary', 'extracted', file_name)
        warped = torch.load(B_path)
        B_path = osp.join('data_binary', 'pose', file_name)
        pose = torch.load(B_path)
        B_path = osp.join('data_binary', 'color', file_name)
        color = torch.load(B_path)

        return {'warped': warped,
                'pose': pose,
                'color': color,
                'label': 10}

    def __len__(self):
        return 10 #len(self.name_list)

dataset_obj = BaseDataset()
sampler = DistributedSampler(dataset_obj, shuffle=False)
train_dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=config.train_batch_size, sampler=sampler)

local_rank = setup()

if local_rank == 0:
    writer = SummaryWriter('runs')

model = MMDiT(depth=args.depth, dim_image= 1152, dim_text = 1152, dim_cond = 1152).to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

if local_rank == 0:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(local_rank)
    vae.requires_grad_(False)
    vae.eval()

@torch.no_grad()
def evaluate(epoch):
    noise = torch.randn([warped.shape[0], 4, 32, 32]).to(local_rank)
    for i in range(0, T)[::-1]:
        t = torch.full((warped.shape[0],), i, device=local_rank).long()
        noise = sample_timestep(noise, t)
    images_t = vae.decode(noise / 0.18215).sample
    writer.add_image('diffusion', torchvision.utils.make_grid(images_t), epoch)

model.train()
global_step = 0

# Now you train the model
for epoch in range(config.num_epochs):
    if local_rank == 0:
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
    else:
        progress_bar = None

    for step, batch in enumerate(train_dataloader):
        warped, pose, cloth = batch["warped"].to(local_rank), batch["pose"].to(local_rank), batch["color"].to(local_rank)
        bs = warped.size(0)
        timesteps = torch.randint(0, T, (bs,), device=warped.device)
        noisy_images, noise = forward_diffusion_sample(warped, timesteps)

        optimizer.zero_grad()
        noise_pred = model(
            image_tokens=noisy_images,
            text_tokens=torch.cat([pose, cloth], dim=1),
            time_cond=timesteps
        )
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if local_rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        global_step += 1
        if local_rank == 0 and global_step % 100 == 0:
            writer.add_scalar("noise", loss.item(), global_step)

    # 🔹 Sync all processes before evaluation

    # 🔹 Only main process runs evaluation — others idle at next barrier
    if local_rank == 0:
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            print("evaluating")
            model.eval()
            with torch.no_grad():
                evaluate(epoch)
            model.train()
            print("evaluating finished \n")

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            torch.save(model.module.state_dict(), config.output_dir)
            print("saved model")