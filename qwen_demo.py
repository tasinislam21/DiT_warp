import spaces
import gradio as gr
import torch
import math
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# Load pipeline with optimized scheduler at startup
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

# Load LoRA for faster inference
pipeline.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
)
pipeline.fuse_lora()


@spaces.GPU(duration=60)
def edit_images(image1, image2, prompt, seed, true_cfg_scale, negative_prompt, num_steps, guidance_scale):
    if image1 is None or image2 is None:
        gr.Warning("Please upload both images")
        return None

    # Convert to PIL if needed
    if not isinstance(image1, Image.Image):
        image1 = Image.fromarray(image1)
    if not isinstance(image2, Image.Image):
        image2 = Image.fromarray(image2)

    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        return output.images[0]


# Example prompts and images
example_prompts = [
    "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
    "Two characters standing side by side in a beautiful garden with flowers blooming",
    "The hero on the left and the villain on the right, facing off in an epic battle scene",
    "Two friends sitting together on a park bench, enjoying the sunset",
]

# Example image paths
example_images = [
    ["bear1.jpg", "bear2.jpg",
     "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."],
]

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # Qwen Image Edit Plus (Optimized)

        Upload two images and describe how you want them combined or edited together.

        [Built with anycoder](https://huggingface.co/spaces/akhaliq/anycoder)
        """
    )

    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(
                label="First Image",
                type="pil",
                height=300
            )
            image2_input = gr.Image(
                label="Second Image",
                type="pil",
                height=300
            )

        with gr.Column():
            output_image = gr.Image(
                label="Edited Result",
                type="pil",
                height=620
            )

    with gr.Group():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Describe how you want the images combined or edited...",
            value=example_prompts[0],
            lines=3
        )

        gr.Examples(
            examples=example_images,
            inputs=[image1_input, image2_input, prompt_input],
            label="Example Images and Prompts"
        )

        gr.Examples(
            examples=[[p] for p in example_prompts],
            inputs=[prompt_input],
            label="Example Prompts Only"
        )

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            seed_input = gr.Number(
                label="Seed",
                value=0,
                precision=0
            )
            num_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=8,
                maximum=30,
                value=8,
                step=1
            )

        with gr.Row():
            true_cfg_scale = gr.Slider(
                label="True CFG Scale",
                minimum=1.0,
                maximum=10.0,
                value=1.0,
                step=0.5
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=5.0,
                value=1.0,
                step=0.1
            )

        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value=" ",
            placeholder="What to avoid in the generation..."
        )

    generate_btn = gr.Button("Generate Edited Image", variant="primary", size="lg")

    generate_btn.click(
        fn=edit_images,
        inputs=[
            image1_input,
            image2_input,
            prompt_input,
            seed_input,
            true_cfg_scale,
            negative_prompt,
            num_steps,
            guidance_scale
        ],
        outputs=output_image,
        show_progress="full"
    )

demo.launch()