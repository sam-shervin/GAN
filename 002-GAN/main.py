import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt


accelerator = Accelerator()
latent_dim = 100
batch_size = 128
lr = 2e-4
epochs = 50
device = accelerator.device


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


class CNNGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512×4×4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            # 512×4×4 → 256×8×8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            # 256×8×8 → 128×16×16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            # 128×16×16 →  3×32×32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [B, latent_dim] → [B, latent_dim,1,1]
        return self.net(z.view(z.size(0), latent_dim, 1, 1))


class CNNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 3×32×32 → 128×16×16
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128×16×16 → 256×8×8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            # 256×8×8 → 512×4×4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            # 512×4×4 → 1×1×1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)


generator     = CNNGenerator()
discriminator = CNNDiscriminator()
optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))
criterion = nn.BCELoss()

generator, discriminator, optim_G, optim_D, dataloader = accelerator.prepare(
    generator, discriminator, optim_G, optim_D, dataloader
)


all_gen = []
for epoch in range(epochs):
    for real, _ in dataloader:
        bs = real.size(0)
        real = real.to(device)
        valid = torch.ones(bs,1,device=device)
        fake  = torch.zeros(bs,1,device=device)

        # G
        optim_G.zero_grad()
        z = torch.randn(bs, latent_dim, device=device)
        gen = generator(z)
        g_loss = criterion(discriminator(gen), valid)
        accelerator.backward(g_loss)
        optim_G.step()

        # D
        optim_D.zero_grad()
        loss_real = criterion(discriminator(real), valid)
        loss_fake = criterion(discriminator(gen.detach()), fake)
        d_loss = 0.5*(loss_real + loss_fake)
        accelerator.backward(d_loss)
        optim_D.step()

    print(f"Epoch {epoch+1}/{epochs}  D: {d_loss:.4f}  G: {g_loss:.4f}")
    with torch.no_grad():
        test_z = torch.randn(16, latent_dim, device=device)
        samples = generator(test_z).cpu()
        all_gen.append(samples)

fig, axs = plt.subplots(10, 17, figsize=(17, 10 * 1.5))

for e in range(-10, 0, 1):
    axs[e, 0].text(0.5, 0.5, f'Epoch {e+1+epochs}', fontsize=10,
                   ha='center', va='center')
    axs[e, 0].axis('off')

    for i in range(16):
        img = all_gen[e][i]                  # (3,32,32)
        img = img.permute(1, 2, 0)           # (32,32,3)
        img = (img + 1) / 2                  # [−1,1] → [0,1]
        axs[e, i+1].imshow(img)
        axs[e, i+1].axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02, hspace=0.1, wspace=0.05)
plt.savefig('gan_generated_epochs_with_labels.png', dpi=900, bbox_inches='tight')
plt.show()
