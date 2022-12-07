import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from ColorizationModel import ColorizationModel
from Dataset import ColorizationDataset
import random
from glob import glob
from tqdm import tqdm


def train_step(imgs, gray_imgs, model, optimizer, criterion):
    model.train()

    colorized_imgs = model(gray_imgs)
    loss = criterion(colorized_imgs, imgs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def val_step(imgs, gray_imgs, model, optimizer, criterion):
    model.eval()
    
    colorized_imgs = model(gray_imgs)
    loss = criterion(colorized_imgs, imgs)

    return loss.item()


def start_train_engine():
    path_to_imagenet_ckpt = "/home/marefat/projects/github/colorization/resnet101-cd907fc2.pth"
    epochs = 100
    batchsize = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"=====> The device is {device}")

    root_path_to_images = "/home/marefat/projects/NSFW_zip_files_dataset/INHOUSE/neutral/*"
    images = [f for f in sorted(glob(root_path_to_images))]

    for _ in range(3):
        random.shuffle(images)

    val_images = images[0: int(0.1*len(images))]
    train_images = images[int(0.1*len(images)): ]

    print(f"=====> Number of train images: {len(train_images)}")
    print(f"=====> Number of valtrain images: {len(val_images)}")

    model = ColorizationModel(path_to_imagenet_pretrained_weights=path_to_imagenet_ckpt)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    
    train_ds = ColorizationDataset(train_images)
    val_ds = ColorizationDataset(val_images, is_train=False)

    train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batchsize, shuffle=True)

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}")

        running_train_loss = []
        for imgs, gray_imgs in tqdm(train_dl):
            imgs, gray_imgs = imgs.to(device), gray_imgs.to(device)

            loss = train_step(imgs, gray_imgs, model, optimizer, criterion)
            running_train_loss.append(loss)
        
        avg_loss_for_train = round(sum(running_train_loss) / len(running_train_loss), 3)


        running_val_loss = []
        for imgs, gray_imgs in val_dl:
            imgs, gray_imgs = imgs.to(device), gray_imgs.to(device)

            loss = val_step(imgs, gray_imgs, model, optimizer, criterion)
            running_val_loss.append(loss)
        
        avg_loss_for_val = round(sum(running_val_loss) / len(running_val_loss), 3)

        with open("./train_report.txt", "a+") as h:
            h.seek(0)
            h.writelines(f"Epoch: {epoch} | AVG Train Loss {avg_loss_for_train} | AVG Val Loss {avg_loss_for_val}\n")
        
        print(f"Epoch: {epoch} | AVG Train Loss {avg_loss_for_train} | AVG Val Loss {avg_loss_for_val}")

        torch.save(model.state_dict(), f"./ckpts/ckpt_{epoch}.pt")
        

if __name__ == "__main__":
    start_train_engine()