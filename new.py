from dataclasses import dataclass
import numpy as np
import torch.utils.data as data
import torch
import os.path as osp
from MMDiT import MMDiT
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from models import FinalLayer
import torchvision
import os

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 1
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

device = accelerator.device

if accelerator.is_main_process:
    writer = SummaryWriter('runs')

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
def sample_timestep(data, cloth_embed, t):
    betas_t = get_index_from_list(betas, t, data.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, data.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, data.shape)
    # Call model (current image - noise prediction)
    with torch.no_grad():
        sample_output = model(data, t, cloth_embed, label)
    model_mean = sqrt_recip_alphas_t * (
            data - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(data)
        posterior_variance_t = get_index_from_list(posterior_variance, t, data.shape)
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


T = 1000
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
        return len(self.name_list)

train_dataloader = torch.utils.data.DataLoader(BaseDataset(), batch_size=config.train_batch_size, shuffle=True)
sample = next(iter(train_dataloader))

model = MMDiT(depth=20)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

if accelerator.is_main_process:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.requires_grad_(False)
    vae.eval()

@torch.no_grad()
def evaluate(epoch):
    noise = torch.randn([warped.shape[0], 4, 32, 32]).to(device)
    cloth_embed, cloth_latent = warper(cloth, pose, label)
    model.eval()
    warper.eval()
    for i in range(0, 1000)[::-1]:
        t = torch.full((1,), i, device=device).long()
        noise = sample_timestep(noise, cloth_embed, t)

    images_t = vae.decode(noise / 0.18215).sample
    ca_warp = vae.decode(cloth_latent / 0.18215).sample

    writer.add_image('diffusion', torchvision.utils.make_grid(images_t), epoch)
    writer.add_image('CA', torchvision.utils.make_grid(ca_warp), epoch)

model, warper, optimizer, optimizer_warp, train_dataloader, lr_scheduler = accelerator.prepare(
    model, warper, optimizer, optimizer_warp, train_dataloader, lr_scheduler
)

global_step = 0

# Now you train the model
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        warped = batch["warped"]
        pose = batch["pose"]
        cloth = batch["color"]
        label = batch["label"]
        # Sample noise to add to the images
        bs = warped.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, 1000, (bs,), device=warped.device,
            dtype=torch.long
        )
        with accelerator.accumulate(warper):
            cloth_embed, cloth_latent = warper(cloth, pose, label)
            loss_warp = F.mse_loss(cloth_latent, warped)
            accelerator.backward(loss_warp)
            optimizer_warp.step()
            optimizer_warp.zero_grad()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        cloth_embed = cloth_embed.detach()
        noisy_images, noise = forward_diffusion_sample(warped, timesteps)
        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, cloth_embed, label)
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        if accelerator.is_main_process and global_step % 100 == 0:
            writer.add_scalar('noise', loss, global_step)
            writer.add_scalar('ca_warp', loss_warp, global_step)

        global_step += 1

    # After each epoch you optionally sample some demo images with evaluate() and save the model
    if accelerator.is_main_process:
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:

            print("Evaluating")
            evaluate(epoch)
            model.train()
            warper.train()
            print("Evaluation Finished")

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            #pipeline.save_pretrained(config.output_dir)
            torch.save(model.state_dict(), os.path.join(config.output_dir, "model.pt"))