"""
flux_eval_block.py — γ ∈ {0.33, 0.5, 1.0} sweep, partitioned for two 4h blocks.

Run twice:
    Block 1: PROMPT_SLICE = (0, 40)
    Block 2: PROMPT_SLICE = (40, 80)

After both blocks done, run flux_metrics.py for final FID.
"""

import os, json
import numpy as np
import torch
import open_clip
from diffusers import FluxPipeline
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch_fidelity

MODEL_PATH    = "/pscratch/sd/a/aditi_12/models/flux-dev-diffusers"
PROFILE_PATH  = "/pscratch/sd/a/aditi_12/projects/flux_profile_dev_1024.npz"
BASE_SAVE_DIR = "/pscratch/sd/a/aditi_12/projects/flux_gamma_sweep"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# >>> CHANGE BETWEEN RUNS <
# PROMPT_SLICE  = (0, 40)
PROMPT_SLICE = (40, 80)
DEVICE         = "cuda"
HEIGHT, WIDTH  = 1024, 1024
GUIDANCE_SCALE = 3.5
REF_STEPS      = 50
NFE_BUDGETS    = [8, 12]
N_BORROW       = 1

LATENT_H, LATENT_W, LATENT_C = HEIGHT//8, WIDTH//8, 16

# γ values to deploy (each gets borrow=1)
GAMMAS = {"gamma033": 0.33, "gamma05": 0.5, "gamma1": 1.0}

# 80 prompts (your original 56 + 24 more — adjust as needed)
PROMPTS = [
    # Nature & Landscapes
    "a bioluminescent bay at midnight with glowing blue waves",
    "a deep sea creature glowing in the dark",
    "a japanese zen garden with a koi pond",
    "a massive thunderstorm over the grand canyon at sunset",
    "a frozen tundra with the northern lights overhead",
    "a dense rainforest waterfall shrouded in morning mist",
    "a volcanic eruption at night reflecting in the ocean",
    "a field of sunflowers under a dramatic storm sky",
    "an ancient redwood forest with shafts of golden light",
    "a desert sand dune landscape at golden hour",
    "a giant panda eating bamboo in a forest",
    "a coral reef teeming with tropical fish and light rays",
    "a himalayan mountain peak above the clouds at sunrise",
    "a lotus pond with evenly spaced flowers, top-down view, uniform lighting",
    "a symmetrical japanese garden with a bridge and cherry blossoms, centered composition, soft lighting",
    # Urban & Architecture
    "a neon-lit cyberpunk alley in tokyo during heavy rain",
    "a grand gothic cathedral interior with stained glass light",
    "a snowy field with one tree, uniform white texture, overcast lighting",
    "a floating market in southeast asia at dawn",
    "a futuristic city skyline reflected in a still river at night",
    "an ancient roman colosseum at golden hour with dramatic clouds",
    "a busy hong kong harbor at night with glowing lights",
    "a medieval castle perched on a foggy cliff over the sea",
    "a moroccan blue city with narrow winding streets",
    "a venetian canal at dusk with gondolas and lanterns",
    "a grand train station with light beams through the roof",
    # Wildlife & Animals
    "a polar bear standing on a melting ice floe at dawn",
    "a pack of wolves running through a snowy pine forest",
    "a humpback whale breaching at sunset over the ocean",
    "a cheetah sprinting across the african savanna",
    "a snow leopard perched on a rocky himalayan cliff",
    "a giant octopus emerging from the deep dark ocean",
    "a flock of flamingos taking flight over a pink lake",
    "a lone wolf howling at a full moon over a frozen lake",
    "a family of elephants crossing a river at golden hour",
    "a majestic eagle diving over a mountain lake",
    # Atmospheric & Mood
    "a lighthouse battered by massive ocean waves in a storm",
    "a cherry blossom tree beside a perfectly still pond",
    "a field of lavender under a purple and pink sunset sky",
    "a dark forest path lit by thousands of fireflies",
    # Block 2 starts here
    "a monastery perched on a sheer cliff face in fog",
    "a cobblestone european village street after heavy rain",
    "a lone fisherman on a misty lake at dawn",
    "a sunken ancient city visible through crystal clear water",
    "a hot air balloon over a patchwork valley at sunrise",
    "a circus tent lit from within on a dark stormy night",
    # Sci-fi & Fantasy
    "an astronaut standing on an alien planet with two moons",
    "a massive spaceship emerging from clouds over a city",
    "a dragon flying over a burning medieval city at night",
    "an underwater city with glowing domes and submarines",
    "a portal to another dimension in the middle of a forest",
    "a giant robot standing in a flooded futuristic city",
    "a wizard casting spells in a lightning-filled stone tower",
    "an ancient temple being reclaimed by a jungle",
    "a steampunk airship fleet navigating through storm clouds",
    "a deep sea exploration vessel near a glowing abyss",
    # Extra prompts to reach 80
    "a peaceful tibetan monastery in fog",
    "a glass cathedral in space",
    "a lighthouse made of crystal under a starlit sky",
    "a ghost ship sailing through arctic ice",
    "an underwater forest with bioluminescent jellyfish",
    "a windmill at sunset over wheat fields",
    "a sushi platter on a polished wooden bar",
    "a winding desert road under a starry sky",
    "an old typewriter on a vintage desk with golden afternoon light",
    "a flock of butterflies in a sunlit meadow",
    "a samurai walking through a misty bamboo forest",
    "a hummingbird hovering over an exotic flower",
    "a farmer leading a horse through golden wheat",
    "an arctic fox in a snowstorm",
    "a glass blower at work in a dim studio",
    "a peacock displaying feathers in a botanical garden",
    "a river otter holding a stone underwater",
    "a Zen tea ceremony in a paper-walled room",
    "a chef plating a beautiful dish in a fine dining kitchen",
    "a mountaineer summiting a peak at dawn",
    "a magnolia tree in full pink bloom against blue sky",
    "an aurora borealis over a frozen pine forest",
    "a city built on the back of a giant turtle",
    "a candlelit study with leather-bound books",
    "a giraffe walking through tall yellow grass at sunset",
]
assert len(PROMPTS) >= 80, f"need ≥80 prompts, got {len(PROMPTS)}"


# =========================================================
# Helpers (lifted directly from your old script)
# =========================================================
def finalize_profile(raw_phi, smooth_sigma=1.0, eps=1e-12):
    raw_phi = np.clip(np.asarray(raw_phi, dtype=np.float64), 0.0, None)
    if smooth_sigma > 0:
        raw_phi = gaussian_filter1d(raw_phi, sigma=smooth_sigma)
    s = raw_phi.sum()
    return raw_phi/s if s > eps else np.ones_like(raw_phi)/len(raw_phi)

def get_pipeline_sigmas_with_mu(pipe, B, height, width, device):
    image_seq_len = (height//8//2) * (width//8//2)
    mu = 0.5 + (image_seq_len-256)/(4096-256) * (1.16-0.5)
    mu = float(np.clip(mu, 0.5, 1.16))
    pipe.scheduler.set_timesteps(num_inference_steps=B, device=device, mu=mu)
    return pipe.scheduler.sigmas.cpu().numpy().astype(np.float32)

def build_shifted_sigmas(B, sigma_start, sigma_end, alpha=3.0):
    t = np.linspace(sigma_start, sigma_end, B+1)
    t = (alpha*t)/(1+(alpha-1)*t+1e-8)
    t[0]=sigma_start; t[-1]=sigma_end
    return t.astype(np.float32)

def build_sigmas_from_phi(phi_desc, sigma_mid_desc, B, sigma_start, sigma_end):
    phi_desc = np.asarray(phi_desc, dtype=np.float64)
    s = phi_desc.sum()
    p_desc = phi_desc/s if s > 0 else np.ones_like(phi_desc)/len(phi_desc)
    p_asc = p_desc[::-1].copy()
    sigma_mid_asc = sigma_mid_desc[::-1].copy()
    cdf_asc = np.cumsum(p_asc); cdf_asc /= cdf_asc[-1]
    sigma_aug = np.concatenate([[sigma_end], sigma_mid_asc, [sigma_start]])
    cdf_aug   = np.concatenate([[0.0], cdf_asc, [1.0]])
    u = np.linspace(0.0, 1.0, B+1)
    sigmas_desc = np.interp(u, cdf_aug, sigma_aug)[::-1].copy()
    sigmas_desc[0] = sigma_start; sigmas_desc[-1] = sigma_end
    for i in range(1, len(sigmas_desc)):
        if sigmas_desc[i] > sigmas_desc[i-1]:
            sigmas_desc[i] = sigmas_desc[i-1]
    return np.asarray(sigmas_desc, dtype=np.float32)

def build_sigmas_borrow(phi_desc, sigma_mid_desc, B, sigma_start, sigma_end,
                        pipeline_sigmas, n_borrow):
    sigmas_full = build_sigmas_from_phi(phi_desc, sigma_mid_desc, B, sigma_start, sigma_end)
    if n_borrow > 0:
        borrowed = pipeline_sigmas[-(n_borrow+2):-2]
        sigmas_full[-(n_borrow+1):-1] = borrowed
    return sigmas_full.astype(np.float32)

@torch.no_grad()
def run_custom_euler(pipe, prompt, sigmas, init_latents):
    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=DEVICE,
        num_images_per_prompt=1, max_sequence_length=256,
    )
    latents = pipe._pack_latents(init_latents.clone(), 1, LATENT_C, LATENT_H, LATENT_W)
    latent_image_ids = pipe._prepare_latent_image_ids(1, LATENT_H, LATENT_W, DEVICE, torch.bfloat16)
    sigmas_t = torch.tensor(sigmas, dtype=torch.bfloat16, device=DEVICE)
    for i in range(len(sigmas_t)-1):
        t, t_next = sigmas_t[i], sigmas_t[i+1]
        v = pipe.transformer(
            hidden_states=latents, timestep=t.reshape(1),
            guidance=torch.tensor([GUIDANCE_SCALE], device=DEVICE, dtype=torch.bfloat16),
            pooled_projections=pooled_embeds.to(torch.bfloat16),
            encoder_hidden_states=prompt_embeds.to(torch.bfloat16),
            txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False,
        )[0]
        latents = latents + (t_next-t)*v
    latents = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents = (latents/pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.image_processor.postprocess(
        pipe.vae.decode(latents, return_dict=False)[0], output_type="pil")[0]

@torch.no_grad()
def get_clip_score(img, prompt):
    img_t = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    txt_t = clip_tokenizer([prompt]).to(DEVICE)
    img_f = clip_model.encode_image(img_t)
    txt_f = clip_model.encode_text(txt_t)
    img_f /= img_f.norm(dim=-1, keepdim=True)
    txt_f /= txt_f.norm(dim=-1, keepdim=True)
    return float((img_f*txt_f).sum())

def compute_rmse(img, ref_arr):
    return float(np.sqrt(((ref_arr-np.array(img).astype(np.float32))**2).mean()))

def make_panel(imgs, labels, prompt, save_path):
    w, h = imgs[0].size
    pad, lh, ph = 10, 40, 28
    canvas = Image.new("RGB", (len(imgs)*w+(len(imgs)+1)*pad, h+lh+ph+3*pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), prompt[:90], fill="black")
    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = pad+i*(w+pad)
        canvas.paste(img, (x, pad+ph+lh))
        draw.text((x+6, pad+ph+4), label, fill="black")
    canvas.save(save_path)


# =========================================================
# Load profile, build phi for each γ
# =========================================================
print("Loading profile...")
d = np.load(PROFILE_PATH)
accel_arr  = d["accel_arr"].astype(np.float64)
time_mid   = d["time_mid"].astype(np.float64)
sigma_mid  = time_mid/1000.0
eps = 1e-12
accel_normed = accel_arr/np.clip(accel_arr.sum(axis=1, keepdims=True), eps, None)
mean_local   = accel_normed.mean(axis=0)

PHIS = {
    name: finalize_profile((mean_local + eps) ** g, smooth_sigma=1.0)
    for name, g in GAMMAS.items()
}


# =========================================================
# Load pipeline + CLIP
# =========================================================
print("Loading FLUX-dev...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
clip_model = clip_model.to(DEVICE).eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')


# =========================================================
# References (load if exist, else generate)
# =========================================================
sigmas_ref = get_pipeline_sigmas_with_mu(pipe, REF_STEPS, HEIGHT, WIDTH, DEVICE)
ref_dir = os.path.join(BASE_SAVE_DIR, "reference_images")
os.makedirs(ref_dir, exist_ok=True)

prompts_block = PROMPTS[PROMPT_SLICE[0]:PROMPT_SLICE[1]]
seeds_block   = list(range(PROMPT_SLICE[0]+200, PROMPT_SLICE[1]+200))

print(f"\nBlock {PROMPT_SLICE}: {len(prompts_block)} prompts")

ref_imgs, ref_latents = [], []
for seed, prompt in tqdm(list(zip(seeds_block, prompts_block)), desc="Reference"):
    init_latents = torch.randn(
        1, LATENT_C, LATENT_H, LATENT_W,
        generator=torch.Generator(DEVICE).manual_seed(seed),
        device=DEVICE, dtype=torch.bfloat16,
    )
    ref_latents.append(init_latents)
    ref_path = os.path.join(ref_dir, f"{seed:05d}.png")
    if os.path.exists(ref_path):
        ref_imgs.append(Image.open(ref_path))
    else:
        img = run_custom_euler(pipe, prompt, sigmas_ref, init_latents)
        img.save(ref_path)
        ref_imgs.append(img)


# =========================================================
# Eval loop
# =========================================================
METHOD_ORDER = ["pipeline", "shifted", "gamma033", "gamma05", "gamma1"]
METHOD_LABELS = {
    "pipeline": f"Pipeline",
    "shifted":  "Shifted α=3",
    "gamma033": "γ=0.33",
    "gamma05":  "γ=0.5",
    "gamma1":   "γ=1.0",
}

for B in NFE_BUDGETS:
    print(f"\n{'='*60}\nNFE = {B}, block {PROMPT_SLICE}")

    B_master = os.path.join(BASE_SAVE_DIR, f"B{B}")
    B_fid    = os.path.join(B_master, "_fid")
    for m in ["reference"] + METHOD_ORDER:
        os.makedirs(os.path.join(B_fid, m), exist_ok=True)

    sigmas_pipeline = get_pipeline_sigmas_with_mu(pipe, B, HEIGHT, WIDTH, DEVICE)
    ps_start = float(sigmas_pipeline[0])
    ps_end   = float(sigmas_pipeline[-2])

    # build all method sigmas
    sigmas_by_method = {
        "pipeline": sigmas_pipeline,
        "shifted":  build_shifted_sigmas(B, ps_start, ps_end, alpha=3.0),
        "gamma033": build_sigmas_borrow(PHIS["gamma033"], sigma_mid, B, ps_start, ps_end, sigmas_pipeline, N_BORROW),
        "gamma05":  build_sigmas_borrow(PHIS["gamma05"],  sigma_mid, B, ps_start, ps_end, sigmas_pipeline, N_BORROW),
        "gamma1":   build_sigmas_borrow(PHIS["gamma1"],   sigma_mid, B, ps_start, ps_end, sigmas_pipeline, N_BORROW),
    }

    # resume support: load existing block partial if exists
    block_json = os.path.join(B_master, f"results_block_{PROMPT_SLICE[0]}_{PROMPT_SLICE[1]}.json")
    if os.path.exists(block_json):
        results_B = json.load(open(block_json))
        done_seeds = {r["seed"] for r in results_B}
        print(f"  resume: loaded {len(results_B)} prompts already done")
    else:
        results_B = []
        done_seeds = set()

    for idx, (seed, prompt) in enumerate(tqdm(list(zip(seeds_block, prompts_block)), desc=f"B={B}")):
        if seed in done_seeds:
            continue

        img_ref = ref_imgs[idx]
        init_latents = ref_latents[idx]
        ref_arr = np.array(img_ref).astype(np.float32)

        method_imgs = {
            m: run_custom_euler(pipe, prompt, sigmas_by_method[m], init_latents)
            for m in METHOD_ORDER
        }

        # save per-prompt images
        prompt_dir = os.path.join(B_master, f"prompt_{seed:05d}")
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
        img_ref.save(os.path.join(prompt_dir, "reference.png"))
        img_ref.save(os.path.join(B_fid, "reference", f"{seed:05d}.png"))
        for m, img in method_imgs.items():
            img.save(os.path.join(prompt_dir, f"{m}.png"))
            img.save(os.path.join(B_fid, m, f"{seed:05d}.png"))

        make_panel(
            [img_ref] + [method_imgs[m] for m in METHOD_ORDER],
            ["Reference"] + [METHOD_LABELS[m] for m in METHOD_ORDER],
            prompt, os.path.join(prompt_dir, "panel.png"),
        )

        # metrics
        r = {"seed": seed, "prompt": prompt}
        for m, img in method_imgs.items():
            r[f"rmse_{m}"] = compute_rmse(img, ref_arr)
            r[f"clip_{m}"] = get_clip_score(img, prompt)
        results_B.append(r)

        # incremental save (resume support)
        with open(block_json, "w") as f:
            json.dump(results_B, f, indent=2)

        if (idx + 1) % 5 == 0:
            mean_rmse = {m: np.mean([x[f"rmse_{m}"] for x in results_B]) for m in METHOD_ORDER}
            print(f"  [{idx+1}/{len(prompts_block)}] running RMSE: " +
                  " ".join(f"{m}={mean_rmse[m]:.1f}" for m in METHOD_ORDER))

    print(f"\nBlock {PROMPT_SLICE} B={B} done.\n  saved → {block_json}")


print(f"\n{'='*60}")
print(f"Block {PROMPT_SLICE} fully done.")
print(f"Next: change PROMPT_SLICE = (40, 80) and rerun, then run flux_metrics.py")