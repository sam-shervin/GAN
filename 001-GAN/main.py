import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.get_device_name(0))
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt

accelerator = Accelerator()

latent_dim = 100
batch_size = 128
lr = 2e-4
epochs = 20
device = accelerator.device

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.net(img)

generator = Generator()
discriminator = Discriminator()

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

generator, discriminator, optim_G, optim_D, dataloader = accelerator.prepare(
    generator, discriminator, optim_G, optim_D, dataloader
)

all_gen_images = []

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        optim_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        accelerator.backward(g_loss)
        optim_G.step()

        optim_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        accelerator.backward(d_loss)
        optim_D.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        gen_imgs = generator(z).cpu()
        all_gen_images.append(gen_imgs)

fig, axs = plt.subplots(epochs, 17, figsize=(17, epochs * 1.5))

for e in range(epochs):
    axs[e, 0].text(0.5, 0.5, f'Epoch {e+1}', fontsize=10, ha='center', va='center')
    axs[e, 0].axis('off')

    for i in range(16):
        axs[e, i+1].imshow(all_gen_images[e][i].squeeze(0), cmap='gray')
        axs[e, i+1].axis('off')

plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02, hspace=0.1, wspace=0.05)
plt.savefig('gan_generated_epochs_with_labels.png', dpi=900, bbox_inches='tight')
plt.show()
