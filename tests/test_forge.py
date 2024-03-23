import torch
import gradio as gr
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from modules import scripts


class MyImageGenerationExtension(scripts.Script):
    def title(self):
        return "My Image Generation Extension"

    def show(self):
        return scripts.AlwaysVisible  # Show in both txt2img and img2img tabs

    def ui(self):
        # Define UI elements
        prompt_input = gr.Textbox(label="Prompt", lines=2)
        generate_button = gr.Button("Generate Image")
        output_image = gr.Image(type="pil")

        # Define function to run on button click
        def generate_image(prompt):
            prior = StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16
            )
            decoder = StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16
            )

            prior.enable_model_cpu_offload()
            decoder.enable_model_cpu_offload()

            prior_output = prior(prompt=prompt, height=1024, width=1024, num_images_per_prompt=1)
            decoder_output = decoder(
                image_embeddings=prior_output.image_embeddings.to(torch.float16),
                prompt=prompt,
                output_type="pil",
            )

            return decoder_output.images[0]

        # Link UI elements to function
        generate_button.click(generate_image, inputs=prompt_input, outputs=output_image)

        return [prompt_input, generate_button, output_image]


# Register the extension
scripts.register_script(MyImageGenerationExtension)
