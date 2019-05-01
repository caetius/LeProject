# Torch
import torch.nn as nn
import torch.optim as optim
from load_data import *

# This project
from noise import corrupt_input
from utils import *

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
    parser.add_argument("--ckpt_on", '-load_weights_from_ckpt', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w')
    args = parser.parse_args()

    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    pretrained_weight_name = "./weights/%s/ae_%d.pkl" % (args.corr_type, args.perc_noise)

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
                wandb.log({"Pretraining Loss": running_loss / 2000,
                           "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        ''' Save Trained Model '''
        print('Saving Model after epoch ', epoch)
        if not os.path.exists('./weights/%s' % args.corr_type):
            os.mkdir('./weights/%s' % args.corr_type)
        torch.save(ae.state_dict(), pretrained_weight_name)

    exit(0)


if __name__ == '__main__':
    main()
