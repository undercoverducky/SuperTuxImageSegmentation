import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

import torch.utils.tensorboard as tb
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from datetime import datetime

def get_dense_transform(random_crop=None, color_change=None,random_horizontal_flip=False):
    import torchvision
    transform = []
    if random_crop is not None:
        transform.append(dense_transforms.RandomResizedCrop(random_crop))
    if random_horizontal_flip:
        transform.append(dense_transforms.RandomHorizontalFlip())
    if color_change is not None:
        transform.append(dense_transforms.ColorJitter(brightness=color_change[0], contrast=color_change[1], saturation=color_change[2], hue=color_change[3]))
    transform.append(dense_transforms.ToTensor())
    return dense_transforms.Compose(transform)
def train(args, device=None):
    from os import path
    model = FCN(layers=[64, 128], normalize_input=True)
    class_weights = [1 / freq for freq in DENSE_CLASS_DISTRIBUTION]
    class_weights = torch.Tensor(class_weights)
    print(f"{model}")
    if device is not None:
        print(f"moving fcn to {device}")
        model = model.to(device)
        class_weights.to(device)
    train_logger, valid_logger = None, None
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, f'train{current_time}'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, f'valid{current_time}'), flush_secs=1)

    max_epochs = 80
    goal_iou = 55

    patience = 7
    best_iou = 0
    epochs_without_improvement = 0
    enabled_early_stopping = False
    scheduler_enabled = True
    #try setting the seed, and using adam
    model_save_freq = 5
    print("Started Loading data")
    training_data_loader = load_dense_data("../dense_data/train", transform=get_dense_transform(color_change=(0.3, 0.3, 0.3, 0.3)))  #color_change > (0.2, 0.2, 0.2, 0.2) random_affine=(5, (0.05, 0.05))
    validation_data_loader = load_dense_data("../dense_data/valid", transform=dense_transforms.ToTensor())
    print("Finished loading data")



    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # try making LR bigger, up to 0.01
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    model.train()
    dataset_size = len(training_data_loader)
    print("Starting training loop")
    train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], 0)
    for epoch in range(max_epochs):
        print(f"Training epoch {str(epoch)}")
        i = 0
        train_c_mat = ConfusionMatrix()
        for batch_x, batch_y in training_data_loader:
            if device is not None:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logit = model.forward(batch_x)
            #print("forward pass finished")
            print(logit.shape)
            print(batch_y.shape)
            loss = criterion(logit, batch_y.long())

            train_logger.add_scalar('loss', loss, epoch * dataset_size + i)
            _, pred_labels = torch.max(logit, dim=1)
            #print(pred_labels)
            train_c_mat.add(pred_labels.to("cpu"), batch_y.to("cpu"))

            loss.backward()
            optimizer.step()
            if i == 0:
                log(train_logger, batch_x.to("cpu"), batch_y, logit.to("cpu"), epoch * dataset_size)
            i += 1

        iou = train_c_mat.iou*100
        train_logger.add_scalar('IOU', iou, epoch * dataset_size + i - 1)

        model.eval()
        val_c_mat = ConfusionMatrix()
        with torch.no_grad():
            for batch_x, batch_y in validation_data_loader:
                if device is not None:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                _, predicted = torch.max(logits, 1)
                val_c_mat.add(predicted.to("cpu"), batch_y.to("cpu"))

        iou = val_c_mat.iou*100
        valid_logger.add_scalar('IOU', iou, epoch * dataset_size + i - 1) # better for skewed datasets

        update_best_iou = False
        if iou > goal_iou or (epoch % model_save_freq == 0 and iou > 50 and iou > best_iou):
            print(f"Saving model at {str(iou)}")
            print(f"Saving model at {str(iou)}")
            save_model(model, str(iou).split(".")[0])
            update_best_iou = True

        if scheduler_enabled:
            scheduler.step(iou)
            train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], iou, epoch * dataset_size + i - 1)

        if enabled_early_stopping:
            if iou > best_iou + 0.1:
                update_best_iou = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. No improvement in validation iou for {patience} epochs.")
                break
            print(f"Epochs without improvement at epoch {epoch}: {epochs_without_improvement}")
        if update_best_iou:
            best_iou = iou

    save_model(model, "final")


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log-dir')

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
    train(args, device=device)
