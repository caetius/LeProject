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

# Set random seed for reproducibility
''' Set Random Seed '''
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def main():

    # Parse Args
    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder")
    parser.add_argument("--valid", '--do_validation',type=bool, default=False,
                        help="Perform validation only.", metavar='v')
    parser.add_argument("--perc_noise", '-percentage_of_noise', type=float, default=0.2,
                        help="Percentage of noise to add.", metavar='p')
    parser.add_argument("--corr_type", '-type_of_noise', type=str, default="mask",
                        help="Percentage of noise to add.", metavar='c')
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=True,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb", '-name_of_wandb_proj', type=str, default="le-project",
                        help="Name of WAND Project.", metavar='w1')
    parser.add_argument("--wandb_on", '-is_wand_on', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w2')
    # possible args: 'orig' (Original AE), 'bn' (Batch Normed version of Original)
    parser.add_argument("--model_type", '-model', type=str, default='orig',
                        help="Type of Autoencoder used.", metavar='ae')

    # Pretraining only
    parser.add_argument("--ckpt_on", '-load_weights_from_ckpt', type=bool, default=True,
                        help="Whether to log to w&b.", metavar='ckpt')

    args = parser.parse_args()

    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(file_path, "weights")):
        os.mkdir(os.path.join(file_path, 'weights'))

    pretrained_weight_name = os.path.join(file_path, "weights/%s/ae_%s_%s.pkl" % (args.corr_type, args.model_type, str(args.perc_noise)))
    print(pretrained_weight_name)

    if args.wandb_on:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    # Create model
    if args.ckpt_on:
        ae = create_model("pretrain", ckpt=pretrained_weight_name, verbose=args.verbose, model_type=args.model_type)
    else:
        ae = create_model("pretrain", ckpt=None, verbose=args.verbose, model_type=args.model_type)
    ae.train()

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_image_loader("../ssl_data_96", 32)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae.parameters())

    if args.wandb_on:
        wandb.watch(ae)


    for epoch in range(40):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(loader_unsup, 0):
            inputs = get_torch_vars(inputs)
            noised = corrupt_input(args.corr_type, inputs, args.perc_noise)
            noised = get_torch_vars(noised)
            print("Iteration number: ", i)

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
                imshow(outputs[0].detach())

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                if args.wandb_on:
                    wandb.log({"Pretraining Loss": running_loss / 2000,
                           "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        ''' Save Trained Model '''
        print('Saving Model after epoch ', epoch)
        if not os.path.exists(os.path.join(file_path,'weights/%s' % args.corr_type)):
            os.mkdir(os.path.join(file_path,'weights/%s' % args.corr_type))
        torch.save(ae.state_dict(), pretrained_weight_name)

    exit(0)


if __name__ == '__main__':
    main()
