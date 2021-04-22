import torch
import torch.nn as nn
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import utils

tqdm = partial(tqdm, leave=True, position=0)


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, loss_fn):
    for _, data in enumerate(tqdm(loader, total=len(loader))):
        data, target = (
            data["input_image"].to(config.DEVICE),
            data["target_image"].to(config.DEVICE),
        )

        opt_disc.zero_grad()
        with torch.cuda.amp.autocast():
            target_fake = gen(data)
            D_real = disc(data, target)
            D_fake = disc(data, target_fake.detach())
            D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
            D_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2.0

        D_loss.backward()
        opt_disc.step()

        opt_gen.zero_grad()
        with torch.cuda.amp.autocast():
            D_fake = disc(data, target_fake)
            G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
            l1_loss = l1(target_fake, target) * config.L1_LAMBDA
            G_loss = G_fake_loss + l1_loss

        G_loss.backward()
        opt_gen.step()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    train_dataset = MapDataset(root_dir=config.TRAIN_PATH)
    val_dataset = MapDataset(root_dir=config.VAL_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_gen, opt_disc, L1_loss, loss_fn,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        utils.save_some_examples(gen, val_loader, filename="evaluation")


if __name__ == "__main__":
    main()
