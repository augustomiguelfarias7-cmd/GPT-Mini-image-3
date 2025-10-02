"""
GPT-Mini image 3 — Augusto
Arquivo: gerador_4k_augusto.py
Descrição:
  - Sistema profissional de geração condicionada por texto em resolução 4K (3840x2160).
  - Estratégia: usar um VAE para comprimir imagens em latentes, rodar difusão/UNet nesses latentes e decodificar para RGB em 4K.
  - Oferece dois modos: "latent_full" e "latent_tiled".

Autor: Augusto
"""

from __future__ import annotations
import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

@dataclass
class Config:
    image_width: int = 3840
    image_height: int = 2160
    image_size: Tuple[int, int] = (3840, 2160)
    in_channels: int = 3
    downscale: int = 16
    latent_channels: int = 4
    text_emb_dim: int = 768
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    lr: float = 1e-4
    epochs: int = 1
    diffusion_steps: int = 1000
    sample_steps: int = 50
    save_dir: str = './checkpoints_4k'
    seed: int = 42
    tiled_overlap: int = 8

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

USE_HF_CLIP = False
try:
    from transformers import CLIPModel, CLIPProcessor
    USE_HF_CLIP = True
except Exception:
    USE_HF_CLIP = False

class CLIPTextEncoder:
    def __init__(self, device='cpu'):
        self.device = device
        self.impl = None
        self.proc = None
        self.model = None
        if USE_HF_CLIP:
            try:
                self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
                from transformers import CLIPProcessor
                self.proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                self.impl = 'hf'
            except Exception:
                self.impl = None
        if self.impl is None:
            pass
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.impl == 'hf':
            inputs = self.proc(text=texts, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            return feats.unsqueeze(1)
        else:
            max_len = 64
            arr = np.zeros((len(texts), max_len), dtype=np.float32)
            for i, t in enumerate(texts):
                codes = [ord(c) % 256 for c in t][:max_len]
                arr[i, :len(codes)] = codes
            tns = torch.tensor(arr, device=self.device)
            W = torch.randn((max_len, cfg.text_emb_dim), device=self.device)
            proj = F.normalize(tns @ W, dim=-1)
            return proj.unsqueeze(1)

class VAEEncoder(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, downscale=8, base=64):
        super().__init__()
        layers = []
        c = in_ch
        out = base
        current_down = 1
        while current_down < downscale:
            layers.append(nn.Conv2d(c, out, 4, 2, 1))
            layers.append(nn.ReLU())
            c = out
            out = min(out*2, 512)
            current_down *= 2
        layers.append(nn.Conv2d(c, latent_ch*2, 3, 1, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_ch=4, out_ch=3, downscale=8, base=64):
        super().__init__()
        c = latent_ch
        layers = [nn.Conv2d(c, base, 3, 1, 1), nn.ReLU()]
        current_down = downscale
        out = base
        while current_down > 1:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(out, max(out//2, 32), 3, 1, 1))
            layers.append(nn.ReLU())
            out = max(out//2, 32)
            current_down //= 2
        layers.append(nn.Conv2d(out, out_ch, 3, 1, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class LatentVAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, downscale=8):
        super().__init__()
        self.enc = VAEEncoder(in_ch, latent_ch, downscale)
        self.dec = VAEDecoder(latent_ch, in_ch, downscale)
    def encode(self, x):
        return self.enc(x)
    def decode(self, z):
        return self.dec(z)

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.act = nn.SiLU()
        if ch_in != ch_out:
            self.skip = nn.Conv2d(ch_in, ch_out, 1)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))

class CrossAttention(nn.Module):
    def __init__(self, ch, text_dim):
        super().__init__()
        self.to_k = nn.Linear(text_dim, ch)
        self.to_v = nn.Linear(text_dim, ch)
        self.to_out = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5
    def forward(self, x, text_emb):
        b,c,h,w = x.shape
        q = x.view(b, c, h*w).permute(0,2,1)
        k = self.to_k(text_emb)
        v = self.to_v(text_emb)
        attn = torch.matmul(q, k.permute(0,2,1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1).contiguous().view(b,c,h,w)
        return self.to_out(out) + x

class LatentUNet(nn.Module):
    def __init__(self, latent_ch=4, base_ch=64, text_dim=768):
        super().__init__()
        self.init = nn.Conv2d(latent_ch, base_ch, 3,1,1)
        self.down1 = ResBlock(base_ch, base_ch*2)
        self.down2 = ResBlock(base_ch*2, base_ch*4)
        self.mid = ResBlock(base_ch*4, base_ch*4)
        self.attn = CrossAttention(base_ch*4, text_dim)
        self.up2 = ResBlock(base_ch*6, base_ch*2)
        self.up1 = ResBlock(base_ch*3, base_ch)
        self.out = nn.Conv2d(base_ch, latent_ch, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x, t, text_emb):
        h = self.init(x)
        d1 = self.down1(h)
        d2 = self.down2(self.pool(d1))
        m = self.mid(self.pool(d2))
        m = self.attn(m, text_emb)
        u2 = self.up2(torch.cat([self.up(m), d2], dim=1))
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1))
        return self.out(u1)

class DDPM:
    def __init__(self, steps=1000, device='cpu'):
        self.steps = steps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x0 + om * noise

class LocalImageDataset(Dataset):
    def __init__(self, folder: str, image_size: Tuple[int,int] = cfg.image_size):
        self.files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    self.files.append(os.path.join(root, f))
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), os.path.splitext(os.path.basename(self.files[idx]))[0]

from torchvision.datasets import CIFAR10

def make_dataloader(paths: List[str], batch_size:int=1, max_items_per_ds:Optional[int]=None):
    ds_list = []
    for p in paths:
        if not p:
            ds = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.Resize(cfg.image_size), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)
            ]))
            ds_list.append(ds)
        elif p.startswith('hf:'):
            continue
        else:
            ds_list.append(LocalImageDataset(p))
    combined = ConcatDataset(ds_list)
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader

class Generator4K:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.text_encoder = CLIPTextEncoder(device=self.device)
        self.vae = LatentVAE(in_ch=cfg.in_channels, latent_ch=cfg.latent_channels, downscale=cfg.downscale).to(self.device)
        self.unet = LatentUNet(latent_ch=cfg.latent_channels, base_ch=64, text_dim=cfg.text_emb_dim).to(self.device)
        self.scheduler = DDPM(steps=cfg.diffusion_steps, device=self.device)
        self.optim = torch.optim.AdamW(list(self.vae.parameters()) + list(self.unet.parameters()), lr=cfg.lr)
    def save(self, tag='latest'):
        path = os.path.join(cfg.save_dir, f'gen4k_{tag}.pt')
        torch.save({'vae':self.vae.state_dict(),'unet':self.unet.state_dict(),'opt':self.optim.state_dict()}, path)
        print('[SAVE]', path)
    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ck['vae'])
        self.unet.load_state_dict(ck['unet'])
        if 'opt' in ck:
            try:
                self.optim.load_state_dict(ck['opt'])
            except Exception:
                print('[LOAD] otimizador não carregado')
        print('[LOAD] checkpoint carregado', path)
    def train(self, dataloader: DataLoader, epochs:int=1):
        self.vae.train(); self.unet.train()
        for ep in range(epochs):
            tot = 0.0
            for i, (imgs, caps) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                texts = [str(c) for c in caps]
                cond = self.text_encoder.encode(texts).to(self.device)
                z, mu, logvar = self.vae.encode(imgs)
                b = z.shape[0]
                t = torch.randint(0, self.scheduler.steps, (b,), device=self.device)
                noise = torch.randn_like(z)
                z_noisy = self.scheduler.q_sample(z, t, noise)
                pred = self.unet(z_noisy, t, cond)
                loss_noise = F.mse_loss(pred, noise)
                rec = self.vae.decode(z)
                loss_rec = F.mse_loss(rec, imgs)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_noise + 0.1*loss_rec + 0.001*kld
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                tot += loss.item()
                if i % 10 == 0:
                    print(f'[TRAIN] ep {ep+1} batch {i+1} loss {tot/(i+1):.4f} (noise {loss_noise.item():.4f})')
            self.save(tag=f'ep{ep+1}')
    @torch.no_grad()
    def sample_full(self, prompts: List[str], steps:int=50):
        cond = self.text_encoder.encode(prompts).to(self.device)
        b = len(prompts)
        h = cfg.image_height // cfg.downscale
        w = cfg.image_width // cfg.downscale
        lat = torch.randn((b, cfg.latent_channels, h, w), device=self.device)
        for s in range(steps-1, -1, -1):
            t = torch.full((b,), int(s*(self.scheduler.steps//steps)), device=self.device, dtype=torch.long)
            pred_noise = self.unet(lat, t, cond)
            beta = self.scheduler.betas[t[0]]
            alpha = self.scheduler.alphas[t[0]]
            lat = (1/torch.sqrt(alpha)) * (lat - (beta/torch.sqrt(1-self.scheduler.alphas_cumprod[t[0]]))*pred_noise)
            if s>0:
                lat = lat + torch.sqrt(beta) * torch.randn_like(lat)
        imgs = self.vae.decode(lat)
        out = []
        to_pil = transforms.ToPILImage()
        for i in range(imgs.shape[0]):
            im = imgs[i].cpu().clamp(-1,1)
            im = (im+1)/2
            out.append(to_pil(im))
        return out
    @torch.no_grad()
    def sample_tiled(self, prompts: List[str], steps:int=50, tile_lat_h:int=64, tile_lat_w:int=64, overlap:int=8):
        cond = self.text_encoder.encode(prompts).to(self.device)
        b = len(prompts)
        H_lat = cfg.image_height // cfg.downscale
        W_lat = cfg.image_width // cfg.downscale
        lat_canvas = torch.zeros((b, cfg.latent_channels, H_lat, W_lat), device=self.device)
        weight_canvas = torch.zeros_like(lat_canvas)
        y_steps = list(range(0, H_lat, tile_lat_h - overlap))
        x_steps = list(range(0, W_lat, tile_lat_w - overlap))
        for y in y_steps:
            for x in x_steps:
                h0 = y; h1 = min(y + tile_lat_h, H_lat)
                w0 = x; w1 = min(x + tile_lat_w, W_lat)
                th = h1 - h0; tw = w1 - w0
                tile = torch.randn((b, cfg.latent_channels, th, tw), device=self.device)
                for s in range(steps-1, -1, -1):
                    t = torch.full((b,), int(s*(self.scheduler.steps//steps)), device=self.device, dtype=torch.long)
                    pred = self.unet(tile, t, cond)
                    beta = self.scheduler.betas[t[0]]
                    alpha = self.scheduler.alphas[t[0]]
                    tile = (1/torch.sqrt(alpha)) * (tile - (beta/torch.sqrt(1-self.scheduler.alphas_cumprod[t[0]]))*pred)
                    if s>0:
                        tile = tile + torch.sqrt(beta) * torch.randn_like(tile)
                wy = torch.linspace(-math.pi/2, math.pi/2, steps=th, device=self.device).unsqueeze(1)
                wx = torch.linspace(-math.pi/2, math.pi/2, steps=tw, device=self.device).unsqueeze(0)
                wy = (torch.cos(wy)+1)/2
                wx = (torch.cos(wx)+1)/2
                blend = (wy * wx).unsqueeze(0).unsqueeze(0)
                lat_canvas[:,:,h0:h1,w0:w1] += tile * blend
                weight_canvas[:,:,h0:h1,w0:w1] += blend
        eps = 1e-6
        lat_canvas = lat_canvas / (weight_canvas + eps)
        imgs = self.vae.decode(lat_canvas)
        out = []
        to_pil = transforms.ToPILImage()
        for i in range(imgs.shape[0]):
            im = imgs[i].cpu().clamp(-1,1)
            im = (im+1)/2
            out.append(to_pil(im))
        return out

def parse_args():
    p = argparse.ArgumentParser(description='Gerador 4K Augusto')
    p.add_argument('--mode', choices=['train','sample','save','load'], default='sample')
    p.add_argument('--datasets', type=str, default='', help='lista de pastas separadas por vírgula')
    p.add_argument('--prompt', type=str, default='um robô heróico no topo da montanha')
    p.add_argument('--out', type=str, default='./outputs')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--epochs', type=int, default=cfg.epochs)
    p.add_argument('--batch', type=int, default=cfg.batch_size)
    p.add_argument('--sample_mode', type=str, default='tiled', choices=['full','tiled'])
    p.add_argument('--tile_lat_size', type=int, default=64, help='tile size in latent pixels (tiled mode)')
    p.add_argument('--steps', type=int, default=cfg.sample_steps)
    return p.parse_args()

def main():
    args = parse_args()
    gen = Generator4K(cfg)
    if args.ckpt:
        gen.load(args.ckpt)
    if args.mode == 'train':
        paths = [p.strip() for p in args.datasets.split(',') if p.strip()]
        dl = make_dataloader(paths, batch_size=args.batch)
        gen.train(dl, epochs=args.epochs)
    elif args.mode == 'sample':
        if args.sample_mode == 'full':
            imgs = gen.sample_full([args.prompt], steps=args.steps)
        else:
            imgs = gen.sample_tiled([args.prompt], steps=args.steps, tile_lat_h=args.tile_lat_size, tile_lat_w=args.tile_lat_size, overlap=cfg.tiled_overlap)
        os.makedirs(args.out, exist_ok=True)
        for i, im in enumerate(imgs):
            path = os.path.join(args.out, f'sample_{i}.png')
            im.save(path)
            print('[OUT]', path)
    elif args.mode == 'save':
        gen.save(tag='manual')
    elif args.mode == 'load':
        if args.ckpt:
            gen.load(args.ckpt)
        else:
            print('Forneça --ckpt')

if __name__ == '__main__':
    main()
