import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

prompt = "an image of a smiley, donning a spacesuit and helmet"

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    num_images_per_prompt=2,
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.to(torch.float16),
    prompt=prompt,
    output_type="pil",
).images[0]
decoder_output.save("cascade.png")
