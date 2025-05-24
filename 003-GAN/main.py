import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(B, -1, W * H)  # B x C' x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W * H)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x
        return out


class CNNGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512×4×4
            nn.utils.spectral_norm(nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False)),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            # Add Self-Attention here for 512 channels
            SelfAttention(512),

            # 512×4×4 → 256×8×8
            nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            # 256×8×8 → 128×16×16
            nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            # 128×16×16 →  3×32×32
            nn.utils.spectral_norm(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class CNNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 3×32×32 → 128×16×16
            nn.utils.spectral_norm(nn.Conv2d(3, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 128×16×16 → 256×8×8
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 256×8×8 → 512×4×4
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Add Self-Attention here
            SelfAttention(512),

            # 512×4×4 → 1×1×1
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, img):
        return self.net(img).view(-1)


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


latent_dim = 128
batch_size = 128
lr_G = 2e-4
lr_D = 1e-4
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

generator = CNNGenerator(latent_dim)
discriminator = CNNDiscriminator()

optim_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.0, 0.9))  # TTUR and recommended for WGAN-GP
optim_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

accelerator = Accelerator()
generator, discriminator, optim_G, optim_D, dataloader = accelerator.prepare(
    generator, discriminator, optim_G, optim_D, dataloader
)


def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


def d_loss_fn(real_pred, fake_pred):
    loss_real = torch.mean(F.relu(1.0 - real_pred))
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    return loss_real + loss_fake


def g_loss_fn(fake_pred):
    return -torch.mean(fake_pred)


all_gen = []
for epoch in range(epochs):
    for real, _ in dataloader:
        bs = real.size(0)
        real = real.to(accelerator.device)
        z = torch.randn(bs, latent_dim, device=accelerator.device)

        # D
        for _ in range(5):  # 5 D steps per G step (common for WGAN-GP)
            optim_D.zero_grad()
            fake = generator(z).detach()
            real_pred = discriminator(real)
            fake_pred = discriminator(fake)
            gp = gradient_penalty(discriminator, real, fake, accelerator.device)
            d_loss = d_loss_fn(real_pred, fake_pred) + 10 * gp  # lambda=10

            accelerator.backward(d_loss)
            optim_D.step()

        # G
        optim_G.zero_grad()
        gen = generator(z)
        fake_pred = discriminator(gen)
        g_loss = g_loss_fn(fake_pred)
        accelerator.backward(g_loss)
        optim_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    with torch.no_grad():
        test_z = torch.randn(16, latent_dim, device=accelerator.device)
        samples = generator(test_z).cpu()
        all_gen.append(samples)


fig, axs = plt.subplots(10, 17, figsize=(17, 10 * 1.5))
for e in range(-10, 0, 1):
    axs[e, 0].text(0.5, 0.5, f'Epoch {e+1+epochs}', fontsize=10,
                   ha='center', va='center')
    axs[e, 0].axis('off')
    for i in range(16):
        img = all_gen[e][i].permute(1, 2, 0)  
        img = (img + 1) / 2  
        axs[e, i+1].imshow(img)
        axs[e, i+1].axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02, hspace=0.1, wspace=0.05)
plt.savefig('wgan_gp_cifar10_generated.png', dpi=300, bbox_inches='tight')
plt.show()
