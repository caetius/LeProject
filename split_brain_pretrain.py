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
    parser.add_argument("--num_classes_ch1", '-num_1', type=int, default=100,
                        help="num classes for single channel: L or r", metavar='ch1')
    parser.add_argument("--num_classes_ch2", '-num_2', type=int, default=10,
                        help="num classes for paired channels: ab or gb", metavar='ch2')
    parser.add_argument("--downsample_size", '-num_pixels', type=int, default=12,
                        help="size of image on which to perform classification", metavar='dsc')


    # Things that change when you put model on GPU or remote cluster
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=True,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb_on", '-is_wand_on', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w2')
    parser.add_argument("--ckpt_on", '-load_weights_from_ckpt', type=bool, default=False,
                        help="Whether to load an existing pretrained ckpt, usually to debug.", metavar='ckpt')
    parser.add_argument("--batch_size", '-num_examples_per_batch', type=int, default=32,
                        help="Batch size.", metavar='bs')

    # Things that change the most
    parser.add_argument("--model_type", '-model', type=str, default='resnet',
                        help="Type of Autoencoder used.", metavar='mod')
    parser.add_argument("--lr_decay", '-learning_rate_decay', type=float, default=0.5,
                        help="percentage by which the learning rate will decrease after every epoch", metavar='lrd')
    # Possible: rgb, lab, lab_distort
    parser.add_argument("--image_space", '-type_of_img_rep', type=str, default="rgb",
                        help="The image space of the input and output of the network.", metavar='ims')

    args = parser.parse_args()

    #### Files ################################################################################################################
    ''' IMPORTANT: Name the weights file and folder such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(file_path, args.weights_folder)):
        os.mkdir(os.path.join(file_path, args.weights_folder))

    ''' Path to save images'''
    if not os.path.exists(os.path.join(file_path, "images")):
        os.mkdir(os.path.join(file_path, 'images'))

    pretrained_weight_name = os.path.join(file_path, "%s/sb_%s_%s.pth" % (args.weights_folder, args.model_type, args.image_space))

    #### Setup #################################################################################################################

    if args.wandb_on:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    print("\n\nStarting Split-Brain... Please, wait while the models load.\n\n")

    # Create model
    if args.ckpt_on:
        split_brain = create_sb_model(type=args.model_type, ckpt=pretrained_weight_name, num_ch2=args.num_classes_ch2, num_ch1=args.num_classes_ch1)
    else:
        split_brain = create_sb_model(type=args.model_type, num_ch2=args.num_classes_ch2, num_ch1=args.num_classes_ch1)

    split_brain.train() # set model to training mode (redundant)

    # Size of model
    pytorch_total_params = sum(p.numel() for p in split_brain.parameters() if p.requires_grad)
    print("\n\nThe model has loaded: Total ", pytorch_total_params, " parameters.")

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_lab_loader("../ssl_data_96", args.batch_size, downsample_params=[args.downsample_size, args.num_classes_ch2, args.num_classes_ch1], image_space=args.image_space)

    # Define an optimizer (with LR rate decay) and criterion
    criterion_ch2 = nn.CrossEntropyLoss()
    criterion_ch1 = nn.CrossEntropyLoss()

    optimizer = optim.Adam(split_brain.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    #### Train #################################################################################################################

    for epoch in range(args.epochs):
        running_loss_ch2 = 0.0
        running_loss_ch1 = 0.0
        for i, (inputs, _, downsample) in enumerate(loader_unsup, 0):
            inputs = get_torch_vars(inputs.type(torch.FloatTensor))
            ch1 = inputs[:,0,:,:] # one channel
            ch2 = inputs[:,1:3,:,:] # two channels

            # ============ Forward ============
            ch2_hat, ch1_hat = split_brain((ch2, ch1))

            #===== Additional Processing For Pixel CrossEntropy =====
            # Quantized labels from resized original image
            # Combine a and b dims to generate 625 classes
            ch2_labels = (downsample[:, 1, :, :] * args.num_classes_ch2 + downsample[:, 2, :, :]).type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor).view(args.batch_size, args.downsample_size**2)
            ch2_labels_unbind = torch.unbind(ch2_labels)
            ch1_labels = downsample[:, 0, :, :].type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor).view(args.batch_size, args.downsample_size**2)
            ch1_labels_unbind = torch.unbind(ch1_labels)
            #print_info("ch2_labels: ", ch2_labels)
            #print_info("ch1_labels: ", ch1_labels)

            # ==== Get predictions for each color class and channel  =====
            ch2_hat_4loss = ch2_hat.permute(0,2,3,1).contiguous().view(args.batch_size, args.downsample_size**2, args.num_classes_ch2**2) #[batch_size*16^2, n_classes_ch2]
            ch2_hat_unbind = torch.unbind(ch2_hat_4loss)

            ch1_hat_4loss = ch1_hat.permute(0,2,3,1).contiguous().view(args.batch_size, args.downsample_size**2, args.num_classes_ch1)    #[batch*256, n_classes_ch1]
            ch1_hat_unbind = torch.unbind(ch1_hat_4loss)

            # ============ Compute Loss ===========
            loss_ch2 = 0.
            loss_ch1 = 0.
            for idx in range(len(ch1_hat_unbind)):
                loss_ch2 += criterion_ch2(ch2_hat_unbind[idx], ch2_labels_unbind[idx])
                loss_ch1 += criterion_ch1(ch1_hat_unbind[idx], ch1_labels_unbind[idx])

            loss = loss_ch1 + loss_ch2

            # ============ Backward ===========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Verbose ============
            '''if args.verbose and i % 1 == 0:

                # Recover output of network as images: Use indices of top-1 logit to identify bins
                ch2_top = torch.topk(ch2_hat_4loss.view(-1, args.num_classes_ch2**2), k=1, dim=1)[1]
                ch1_top = torch.topk(ch1_hat_4loss.view(-1, args.num_classes_ch1), k=1, dim=1)[1]
                print_info("ch1 top_1_indices: ", ch1_top)
                print_info("ch2 top_1_indices: ", ch2_top)

                # Get two dimensions of color classification bins (625 bins)->(25 bins,25 bins)
                ch2_images = ch2_top.view(args.batch_size, 1, args.downsample_size, args.downsample_size)
                ch2_images = torch.cat((ch2_images % args.num_classes_ch2, ch2_images / args.num_classes_ch2), 1)

                # Convert to Float tensor to denormalise
                ch2_images = (ch2_images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))

                # Reshape ch1 tensor
                ch1_images = ch1_top.view(args.batch_size, 1, args.downsample_size, args.downsample_size).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                # Get full image tensor [batch, 3, downsample_size, downsample_size]
                reconstructed = torch.cat((ch1_images, ch2_images), 1).detach().cpu()

                # Get images back as RGB for display
                if args.image_space == "rgb":
                    rgb_input = rgb_to_list(inputs.cpu())
                    rgb_output = rgb_to_list(rescale_color("rgb", reconstructed, args.num_classes_ch1, args.num_classes_ch2))
                    rgb_input_from_downsized = rgb_to_list(rescale_color("rgb", downsample, args.num_classes_ch1, args.num_classes_ch2))
                else:
                    rgb_input = lab_to_rgb_list(denormalize_lab(inputs).cpu())
                    rgb_output = lab_to_rgb_list(rescale_color("lab", reconstructed, args.num_classes_ch1, args.num_classes_ch2))
                    rgb_input_from_downsized = lab_to_rgb_list(rescale_color("lab", downsample, args.num_classes_ch1, args.num_classes_ch2))
                #grid_imshow(rgb_input, rgb_output, rgb_input_from_downsized, second_label="Original Downsized")
            '''

            # ============ Logging ============
            running_loss_ch2 += loss_ch2.data
            running_loss_ch1 += loss_ch1.data
            iterations_to_check = 100
            if i % iterations_to_check == 1:
                if args.wandb_on and i > 1:
                    wandb.log({"Pretraining Loss": (running_loss_ch2+running_loss_ch1) / iterations_to_check,
                            "CH2 Loss" : running_loss_ch2 / iterations_to_check,
                            "CH1 Loss": running_loss_ch1 / iterations_to_check,
                            "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, (running_loss_ch2+running_loss_ch1) / iterations_to_check))
                running_loss_ch1 = 0.
                running_loss_ch2 = 0.
        ''' Save Trained Model '''
        print('Saving Model after each epoch ', epoch)
        torch.save(split_brain.state_dict(), pretrained_weight_name)

        ''' Update Learning Rate '''
        scheduler.step()

    exit(0)


if __name__ == '__main__':
    main()