from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.nn.functional as F
import torch.optim as optim
import torch
from datetime import datetime


def get_transform(resize=None, random_crop=None, color_change=None,random_affine=None, random_horizontal_flip=False, random_vertical_flip=False, normalize=False):
    import torchvision

    transform = []
    if resize is not None:
        transform.append(torchvision.transforms.Resize(resize))
    if random_crop is not None:
        transform.append(torchvision.transforms.RandomResizedCrop(random_crop))
    if random_horizontal_flip:
        transform.append(torchvision.transforms.RandomHorizontalFlip())
    if random_vertical_flip:
        transform.append(torchvision.transforms.RandomVerticalFlip())
    if color_change is not None:
        transform.append(torchvision.transforms.ColorJitter(brightness=color_change[0], contrast=color_change[1], saturation=color_change[2], hue=color_change[3]))
    if random_affine is not None:
        transform.append(torchvision.transforms.RandomAffine(degrees=random_affine[0],translate=random_affine[1]))
    transform.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(transform)


def train(args, device=None):
    from os import path
    model = CNNClassifier(layers=[32, 64, 128], normalize_input=True)
    if device is not None:
        model.to(device)
    train_logger, valid_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, f'valid{current_time}'), flush_secs=1)

    max_epochs = 80
    goal_acc = 90

    patience = 15
    best_acc = 0
    epochs_without_improvement = 0
    enabled_early_stopping = False
    scheduler_enabled = True

    model_save_freq = 5

    training_data_loader = load_data("../data/train", transform=get_transform(color_change=(0.2, 0.2, 0.2, 0.2))) # random_affine=(5, (0.05, 0.05))
    validation_data_loader = load_data("../data/valid", transform=get_transform())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    model.train()
    dataset_size = len(training_data_loader)
    print("Starting training loop")
    for epoch in range(max_epochs):
        print(f"Training epoch {str(epoch)}")
        num_correct = 0
        num_samples = 0
        i = 0
        for batch_x, batch_y in training_data_loader:
            if device is not None:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logit = model.forward(batch_x)
            loss = F.cross_entropy(logit, batch_y)
            train_logger.add_scalar('loss', loss, epoch * dataset_size + i)

            _, pred_labels = torch.max(logit, dim=1)
            num_correct += (pred_labels == batch_y).sum().item()
            num_samples += batch_y.size(0)

            loss.backward()
            optimizer.step()
            i += 1

        acc = 100 * num_correct / num_samples
        train_logger.add_scalar('accuracy', acc, epoch * dataset_size + i - 1)

        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in validation_data_loader:
                if device is not None:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                _, predicted = torch.max(logits, 1)
                num_samples += batch_y.size(0)
                num_correct += (predicted == batch_y).sum().item()

        acc = 100 * num_correct / num_samples
        valid_logger.add_scalar('accuracy', acc, epoch * dataset_size + i - 1)
        update_best_acc = False
        if acc > 94 or (epoch % model_save_freq == 0 and acc > goal_acc and acc > best_acc):
            print(f"Saving model at {str(acc)}")
            save_model(model, str(acc).split(".")[0])
            update_best_acc = True

        if scheduler_enabled:
            scheduler.step(acc)

        if enabled_early_stopping:
            if acc > best_acc + 0.1:
                update_best_acc = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. No improvement in validation accuracy for {patience} epochs.")
                break
            print(f"Epochs without improvement at epoch {epoch}: {epochs_without_improvement}")
        if update_best_acc:
            best_acc = acc

    save_model(model, "final")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir')

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif device != "mps":
        device = torch.device('cpu')
    device = torch.device(device)
    print(f"Using device: {device}")

    args = parser.parse_args()
    train(args, device)
