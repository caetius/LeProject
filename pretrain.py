# Torch
import torch.nn as nn
import torch.optim as optim
from load_data import *

# This project
from noise import corrupt_input
from utils import *

# Torchvision
import torchvision

# OS
import os
import argparse

#WANDB
import wandb
wandb.init(project="le-project")


# Set random seed for reproducibility
''' Set Random Seed '''
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def main():
    wandb.init()

    # Parse Args
    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder")
    parser.add_argument("--valid", '--do_validation',type=bool, default=False,
                        help="Perform validation only.", metavar='v')
    parser.add_argument("--perc_noise", '-percentage_of_noise', type=float, default=0.05,
                        help="Percentage of noise to add.", metavar='p')
    parser.add_argument("--corr_type", '-type_of_noise', type=str, default="sp",
                        help="Percentage of noise to add.", metavar='c')
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=False,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb", '-name_of_wandb_proj', type=str, default="le-project",
                        help="Name of WAND Project.", metavar='w')
    args = parser.parse_args()

    wandb.config.update(args)


    # Create model
    ae = create_model("pretrain")

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_image_loader("../ssl_data_96", 32)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae.parameters())

    wandb.watch(ae)

    for epoch in range(30):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(loader_unsup, 0):
            inputs = get_torch_vars(inputs)
            noised = corrupt_input(args.corr_type, inputs, args.perc_noise)
            noised = get_torch_vars(noised)

            # ============ Forward ============
            encoded, outputs = ae(noised)
            loss = criterion(outputs, inputs)
            # ============ Backward ===========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ============ Verbose ============
            if args.verbose:
                imshow(inputs[0])
                imshow(noised[0])

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                wandb.log({"Training Loss": running_loss / 2000,
                           "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        ''' Save Trained Model '''
        print('Saving Model after epoch ', epoch)
        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        torch.save(ae.state_dict(), "./weights/ae.pkl")

    ''' Do Validation '''
    if args.valid:
        print("Loading checkpoint...")
        ae.load_state_dict(torch.load("./weights/ae.pkl"))
        dataiter = iter(loader_val_sup)
        images, labels = dataiter.next()
        # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        images = Variable(images.cuda())
        decoded_imgs = ae(images)[1]
        if args.verbose:
            imshow(torchvision.utils.make_grid(images))
            imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)


if __name__ == '__main__':
    main()
