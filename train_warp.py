from dataclasses import dataclass
import torch.utils.data as data
import torch
import os.path as osp
from models import WarpAdapter
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import torchvision
import os

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 32
    num_epochs = 1000
    save_image_epochs = 10
    output_dir = "ddpm-butterflies-128"
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision

config = TrainingConfig()

accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=1,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

device = accelerator.device

if accelerator.is_main_process:
    writer = SummaryWriter('runs')

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

warper = WarpAdapter()
optimizer_warp = torch.optim.AdamW(warper.parameters(), lr=1e-4)

if accelerator.is_main_process:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.requires_grad_(False)
    vae.eval()

@torch.no_grad()
def evaluate(epoch):
    ca_warp = vae.decode(cloth_latent / 0.18215).sample
    writer.add_image('CA', torchvision.utils.make_grid(ca_warp), epoch)

warper, optimizer_warp, train_dataloader = accelerator.prepare(warper, optimizer_warp, train_dataloader)

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



        progress_bar.update(1)
        logs = {"loss": loss_warp.detach().item(), "step": global_step}
        progress_bar.set_postfix(**logs)
        if accelerator.is_main_process and global_step % 100 == 0:
            writer.add_scalar('ca_warp', loss_warp, global_step)

        global_step += 1

    # After each epoch you optionally sample some demo images with evaluate() and save the model
    if accelerator.is_main_process:
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:

            print("Evaluating")
            evaluate(epoch)
            print("Evaluation Finished")

        if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
            #pipeline.save_pretrained(config.output_dir)
            torch.save(warper.state_dict(), os.path.join(config.output_dir, "model.pt"))