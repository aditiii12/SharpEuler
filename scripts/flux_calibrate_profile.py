# paste above
import torch
import numpy as np
from diffusers import FluxPipeline
from tqdm import tqdm

# run_calibrate_dev.py
MODEL_PATH = "/pscratch/sd/a/aditi_12/models/flux-dev-diffusers"
SAVE_PATH  = "/pscratch/sd/a/aditi_12/projects/flux_profile_dev_1024.npz"
N_STEPS    = 50   # dev works well at 20-50 steps
N_PROMPTS  = 64
DEVICE     = "cuda"
HEIGHT, WIDTH = 1024, 1024

PROMPTS = [
    "a red apple on a wooden table",
    "a dog running on the beach at sunset",
    "a futuristic city skyline at night",
    "a bowl of fresh fruit in natural light",
    "a mountain landscape with snow",
    "a close-up portrait of an elderly man",
    "a cat sitting on a windowsill",
    "a colorful hot air balloon over fields",
    "a plate of spaghetti with tomato sauce",
    "a sailboat on calm blue water",
    "a forest path in autumn",
    "a child playing in the rain",
    "a coffee cup on a wooden desk",
    "a lion resting in the savanna",
    "a waterfall in a tropical jungle",
    "a vintage car on a country road",
    "a snowy village at christmas",
    "a surfer riding a large wave",
    "a field of sunflowers at golden hour",
    "a cozy fireplace in a cabin",
    "a robot in a futuristic laboratory",
    "a woman reading a book in a garden",
    "a colorful parrot in a rainforest",
    "a pizza fresh out of the oven",
    "a lighthouse on a rocky coast",
    "a dragon flying over mountains",
    "a ballet dancer on stage",
    "a steam train through countryside",
    "an astronaut floating in space",
    "a zen garden with stones and sand",
    "a tiger drinking from a river",
    "a wedding ceremony in a church",
    "a chef cooking in a restaurant kitchen",
    "a bicycle on a cobblestone street",
    "a thunderstorm over the ocean",
    "a polar bear on an ice floe",
    "a medieval castle on a hill",
    "a market stall with exotic spices",
    "a horse galloping through a meadow",
    "a neon-lit street in tokyo at night",
    "a glass of red wine on a table",
    "a peacock displaying its feathers",
    "a canoe on a misty lake",
    "a bee collecting pollen from a flower",
    "a grand piano in a concert hall",
    "a submarine underwater",
    "a campfire under the stars",
    "a golden retriever playing fetch",
    "a cactus in the desert at dawn",
    "a bowl of ramen with chopsticks",
    "an old library with wooden shelves",
    "a speedboat on a river",
    "a swan on a still lake",
    "a skier on a snowy slope",
    "a tropical beach with palm trees",
    "a ferris wheel at an amusement park",
    "a wolf howling at the moon",
    "a cathedral interior with stained glass",
    "a bunch of colorful balloons",
    "a fox in a snowy forest",
    "a surreal painting of melting clocks",
    "a busy new york city intersection",
    "a rose covered in morning dew",
    "a samurai in traditional armor",
]
N_PROMPTS = len(PROMPTS)  # 64 unique prompts

print(f"Loading FLUX pipeline...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

# ── hook scheduler.step to capture velocity fields ────────
captured_fields = []
captured_times  = []
original_step   = pipe.scheduler.step

def hooked_step(model_output, timestep, sample, **kwargs):
    captured_fields.append(model_output.detach().float().cpu())
    captured_times.append(float(timestep))
    return original_step(model_output, timestep, sample, **kwargs)

pipe.scheduler.step = hooked_step

# ── calibration loop ──────────────────────────────────────
all_accel_norms = []
all_kappa_geom  = []
time_steps_global = None
time_mid_global   = None
eps = 1e-8

print(f"Running calibration: {N_PROMPTS} prompts x {N_STEPS} steps")

# sanity check on first prompt
first_prompt = True

for j, prompt in enumerate(tqdm(PROMPTS, desc="Calibration")):
    captured_fields.clear()
    captured_times.clear()

    with torch.no_grad():
        _ = pipe(
            prompt=prompt,
            num_inference_steps=N_STEPS,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(DEVICE).manual_seed(j),
        )

    fields = captured_fields.copy()
    times  = captured_times.copy()

    if len(fields) < 2:
        raise RuntimeError(f"Prompt {j}: expected ≥2 fields, got {len(fields)}")

    # sanity check on first prompt
    if first_prompt:
        print(f"\nSanity check (prompt 0):")
        print(f"  n_steps captured: {len(fields)}")
        print(f"  field shape: {fields[0].shape}")
        print(f"  field magnitude: {fields[0].norm():.4f}")
        print(f"  time range: {times[0]:.4f} → {times[-1]:.4f}")
        first_prompt = False

    if time_steps_global is None:
        time_steps_global = np.array(times, dtype=np.float64)
        time_mid_global   = 0.5 * (time_steps_global[:-1] + time_steps_global[1:])

    accel_norms_j = []
    kappa_geom_j  = []

    for i in range(len(fields) - 1):
        f1 = fields[i].flatten().float()
        f2 = fields[i+1].flatten().float()

        dt = max(abs(times[i+1] - times[i]), eps)
        a_hat = (f2 - f1) / dt

        f1_sq_mean  = (f1 * f1).mean().clamp_min(eps)
        a_dot_f     = (a_hat * f1).mean()
        a_par       = (a_dot_f / f1_sq_mean) * f1
        a_orth      = a_hat - a_par

        accel       = torch.sqrt((a_hat   * a_hat  ).mean())
        f1_rms      = torch.sqrt(f1_sq_mean)
        a_orth_rms  = torch.sqrt((a_orth  * a_orth ).mean())
        kappa       = a_orth_rms / (f1_rms * f1_rms + eps)

        accel_norms_j.append(float(accel))
        kappa_geom_j.append(float(kappa))

    all_accel_norms.append(accel_norms_j)
    all_kappa_geom.append(kappa_geom_j)

# ── save ──────────────────────────────────────────────────
accel_arr  = np.array(all_accel_norms, dtype=np.float64)
kappa_arr  = np.array(all_kappa_geom,  dtype=np.float64)
mean_accel = accel_arr.mean(axis=0)
mean_kappa = kappa_arr.mean(axis=0)

np.savez(
    SAVE_PATH,
    mean_accel   = mean_accel,
    mean_kappa   = mean_kappa,
    accel_arr    = accel_arr,
    kappa_arr    = kappa_arr,
    time_steps   = time_steps_global,
    time_mid     = time_mid_global,
    time_start   = time_steps_global[0],
    time_end     = time_steps_global[-1],
)

print(f"\nSaved to {SAVE_PATH}")
print(f"  accel_arr shape : {accel_arr.shape}")
print(f"  time_mid shape  : {time_mid_global.shape}")
print(f"  time range      : {time_steps_global[0]:.4f} → {time_steps_global[-1]:.4f}")
print(f"  accel: min={mean_accel.min():.6f} max={mean_accel.max():.6f} mean={mean_accel.mean():.6f}")
print(f"  kappa: min={mean_kappa.min():.6f} max={mean_kappa.max():.6f} mean={mean_kappa.mean():.6f}")