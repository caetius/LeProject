# Torch
import torch.nn as nn
import torch.optim as optim
from load_data import *

# This project
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

    #### Args ################################################################################################################

    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder")
    # Things that rarely change
    parser.add_argument("--wandb", '-name_of_wandb_proj', type=str, default="le-project",
                        help="Name of WAND Project.", metavar='w1')
    parser.add_argument("--weights_folder", '-folder_name', type=str, default='weights',
                        help="Name of weights folder for all weights.", metavar='w')
    parser.add_argument("--epochs", '-num_epochs', type=int, default=40,
                        help="Number of epochs.", metavar='ep')


    # Things that change when you put model on GPU or remote cluster
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=True,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb_on", '-is_wand_on', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w2')
    parser.add_argument("--ckpt_on", '-load_weights_from_ckpt', type=bool, default=False,
                        help="Whether to load an existing pretrained ckpt, usually to debug.", metavar='ckpt')
    parser.add_argument("--batch_size", '-num_examples_per_batch', type=int, default=2,
                        help="Batch size.", metavar='bs')

    # Things that change the most
    parser.add_argument("--model_type", '-model', type=str, default='simple',
                        help="Type of Autoencoder used.", metavar='ae')
    parser.add_argument("--num_ab_classes", '-num_ab', type=int, default=10,
                        help="num ab classes", metavar='abc')
    parser.add_argument("--num_L_classes", '-num_L', type=int, default=100,
                        help="num ab classes", metavar='abl')
    parser.add_argument("--downsample_size", '-num_pixels', type=int, default=12,
                        help="size of image on which to perform classification", metavar='dsc')
    parser.add_argument("--lr_decay", '-learning_rate_decay', type=int, default=0.5,
                        help="percentage by which the learning rate will decrease after every epoch", metavar='lrd')

    args = parser.parse_args()

    #### Files ################################################################################################################
    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(file_path, args.weights_folder)):
        os.mkdir(os.path.join(file_path, args.weights_folder))

    ''' Path to save images'''
    if not os.path.exists(os.path.join(file_path, "images")):
        os.mkdir(os.path.join(file_path, 'images'))

    pretrained_weight_name = os.path.join(file_path, "%s/sb_%s.pth" % (args.weights_folder, args.model_type))

    #### Setup #################################################################################################################

    if args.wandb_on:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    print("\n\nStarting Split-Brain... Please, wait while the models load.\n\n")

    # Create model
    if args.ckpt_on:
        split_brain = create_sb_model(type=args.model_type, ckpt=pretrained_weight_name, num_ab=args.num_ab_classes, num_L=args.num_L_classes)
    else:
        split_brain = create_sb_model(type=args.model_type, num_ab=args.num_ab_classes, num_L=args.num_L_classes)

    split_brain.train()

    pytorch_total_params = sum(p.numel() for p in split_brain.parameters() if p.requires_grad)


    print("\n\nThe model has loaded: Total ", pytorch_total_params, " parameters.")

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_lab_loader("../ssl_data_96", args.batch_size, downsample_params=[args.downsample_size, args.num_ab_classes, args.num_L_classes])

    # Define an optimizer and criterion
    criterion_ab = nn.CrossEntropyLoss()
    criterion_L = nn.CrossEntropyLoss()

    optimizer = optim.Adam(split_brain.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    #### Train #################################################################################################################

    for epoch in range(args.epochs):
        running_loss_ab = 0.0
        running_loss_L = 0.0
        for i, (inputs, _, downsample) in enumerate(loader_unsup, 0):
            inputs = get_torch_vars(inputs.type(torch.FloatTensor))
            L = inputs[:,0,:,:] # one channel
            ab = inputs[:,1:3,:,:] # two channels

            # ============ Forward ============
            ab_hat, L_hat = split_brain((ab, L))

            #===== Additional Processing For Pixel CrossEntropy =====
            # Quantized labels from resized original image
            # Combine a and b dims to generate 625 classes
            ab_labels = (downsample[:, 1, :, :] * args.num_ab_classes + downsample[:, 2, :, :]).type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor).view(args.batch_size, args.downsample_size**2)
            ab_labels_unbind = torch.unbind(ab_labels)
            L_labels = downsample[:, 0, :, :].type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor).view(args.batch_size, args.downsample_size**2)
            L_labels_unbind = torch.unbind(L_labels)

            # ==== Get predictions for each color class and channel  =====
            ab_hat_4loss = ab_hat.permute(0,2,3,1).contiguous().view(args.batch_size, args.downsample_size**2, args.num_ab_classes**2) #[batch_size*16^2, n_classes_ab]
            ab_hat_unbind = torch.unbind(ab_hat_4loss)

            L_hat_4loss = L_hat.permute(0,2,3,1).contiguous().view(args.batch_size, args.downsample_size**2, args.num_L_classes)    #[batch*256, n_classes_L]
            L_hat_unbind = torch.unbind(L_hat_4loss)

            # ============ Compute Loss ===========
            loss_ab = 0.
            loss_L = 0.
            for idx in range(len(L_hat_unbind)):
                loss_ab += criterion_ab(ab_hat_unbind[idx], ab_labels_unbind[idx])
                loss_L += criterion_L(L_hat_unbind[idx], L_labels_unbind[idx])

            loss = loss_L + loss_ab

            # ============ Backward ===========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Verbose ============
            if args.verbose:
                print("Iteration number: ", i, ", Loss: ", loss.data)
                print("--------------------------------------------------")
                print("L: ", L.shape, ", ab: ", ab.shape, ", downsample: ", downsample.shape)
                print("AB_hat: ", ab_hat.shape, ", L_hat: ", L_hat.shape)
                print("AB Labels: ", ab_labels.shape, ", L Labels: ", L_labels.shape)
                print("AB_hat_4loss: ", ab_hat_4loss.shape, ", L_hat_4loss: ", L_hat_4loss.shape)
                print("TOTAL LOSS: ", loss.data)
                print("--------------------------------------------------")

                # Recover output of network as images: Use indices of top-1 logit to identify bins
                ab_top = torch.topk(ab_hat_4loss.view(-1, args.num_ab_classes**2), k=1, dim=1)[1]
                L_top = torch.topk(L_hat_4loss.view(-1, args.num_L_classes), k=1, dim=1)[1]
                print_info(ab_top)
                print_info(L_top)

                # Get two dimensions of color classification bins (625 bins)->(25 bins,25 bins)
                ab_images = ab_top.view(args.batch_size, 1, args.downsample_size, args.downsample_size)
                ab_images = torch.cat((ab_images % args.num_ab_classes, ab_images / args.num_ab_classes), 1)

                # Convert to Float tensor to denormalise
                ab_images = (ab_images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))

                # Reshape L tensor
                L_images = L_top.view(args.batch_size, 1, args.downsample_size, args.downsample_size).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                # Get full Lab tensor [batch, 3, downsample_size, downsample_size]
                reconstructed = torch.cat((L_images, ab_images), 1).detach().cpu()

                # Get RGB of images
                rgb_input = lab_to_rgb(denormalize(inputs).cpu())

                # denormalise color range -> (LAB) -> (RGB)
                rgb_output = lab_to_rgb(rescale_color(reconstructed, args.num_ab_classes))

                # denormalise color range -> (LAB) -> (RGB)
                rgb_input_from_downsized = lab_to_rgb(rescale_color(downsample, args.num_ab_classes))
                grid_imshow(rgb_input, rgb_output, rgb_input_from_downsized, second_label="Original Downsized")

            # ============ Logging ============
            running_loss_ab += loss_ab.data
            running_loss_L += loss_L.data
            if i % 2000 == 1999:
                if args.wandb_on:
                    wandb.log({"Pretraining Loss": (running_loss_ab+running_loss_L) / 2000,
                            "AB Loss" : running_loss_ab / 2000,
                            "L Loss": running_loss_L / 2000,
                            "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, (running_loss_ab+running_loss_L) / 2000))
                running_loss_L = 0.
                running_loss_ab = 0.
        ''' Save Trained Model '''
        print('Saving Model after each epoch ', epoch)
        torch.save(split_brain.state_dict(), pretrained_weight_name)


        ''' Update Learning Rate '''
        scheduler.step()

    exit(0)


if __name__ == '__main__':
    main()