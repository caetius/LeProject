from utils import *
from load_data import *
import os
import argparse

import torch.nn as nn
import torch.optim as optim

import wandb

def main():


    #### Args ################################################################################################################

    parser = argparse.ArgumentParser(description="Train Split-Brain Autoencoder")
    # Things that rarely change
    parser.add_argument("--valid", '--do_validation',type=bool, default=True,
                        help="Perform validation only.", metavar='v')
    parser.add_argument("--wandb", '-name_of_wandb_proj', type=str, default="le-project",
                        help="Name of WAND Project.", metavar='w1')
    parser.add_argument("--weights_folder", '-folder_name', type=str, default='weights',
                        help="Name of weights folder for all weights.", metavar='w')
    parser.add_argument("--epochs", '-num_epochs', type=int, default=20,
                        help="Number of epochs.", metavar='ep')
    parser.add_argument("--num_ab_classes", '-num_ab', type=int, default=10,
                        help="num ab classes", metavar='abc')
    parser.add_argument("--num_L_classes", '-num_L', type=int, default=100,
                        help="num ab classes", metavar='abl')
    parser.add_argument("--downsample_size", '-num_pixels', type=int, default=12,
                        help="size of image on which to perform classification", metavar='dsc')

    # Things that change when you put model on GPU or remote cluster
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=True,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb_on", '-is_wand_on', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w2')
    parser.add_argument("--batch_size", '-num_examples_per_batch', type=int, default=32,
                        help="Batch size.", metavar='bs')


    # Things that change the most
    parser.add_argument("--model_type", '-model', type=str, default='alex',
                        help="Type of Autoencoder used.", metavar='ae')
    parser.add_argument("--lr_decay", '-learning_rate_decay', type=float, default=0.5,
                        help="percentage by which the learning rate will decrease after every epoch", metavar='lrd')
    # Possible: rgb, lab, lab_distort
    parser.add_argument("--image_space", '-type_of_img_rep', type=str, default="lab_distort",
                        help="The image space of the input and output of the network.", metavar='ims')

    args = parser.parse_args()


    #### Files ################################################################################################################

    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_weight_name = os.path.join(file_path, "%s/sb_%s_%s.pth" % (args.weights_folder, args.model_type, args.image_space))
    finetuned_weight_name = os.path.join(file_path,"%s/sb_finetuned_%s_%s.pth" % (args.weights_folder, args.model_type, args.image_space))

    ''' Path to save images'''
    if not os.path.exists(os.path.join(file_path, "images")):
        os.mkdir(os.path.join(file_path, 'images'))

    # Checks that the pretrained weight folder exist.
    if not os.path.exists(os.path.join(file_path, args.weights_folder)):
        raise Exception('Your pretrained weights folder is missing')


    #### Setup #################################################################################################################


    if args.wandb_on:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    # Create model
    classifier = create_sb_model(type="classifier_"+args.model_type+"_shallow", ckpt=pretrained_weight_name, num_ab=args.num_ab_classes, num_L=args.num_L_classes)

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_lab_loader("../ssl_data_96", args.batch_size, downsample_params=[args.downsample_size, args.num_ab_classes, args.num_L_classes])

    # Define an optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    if args.wandb_on:
        wandb.watch(classifier)

    prev_top1 = 0.

    #### Train #################################################################################################################


    for epoch in range(args.epochs):
        running_loss = 0.0

        classifier.train()

        for i, (inputs, labels, _) in enumerate(loader_sup, 0):
            inputs = get_torch_vars(inputs.type(torch.FloatTensor))
            L = inputs[:, 0, :, :]  # one channel
            ab = inputs[:, 1:3, :, :]  # two channels
            labels = get_torch_vars(labels)

            optimizer.zero_grad()

            # ============ Forward ============
            out = classifier((ab, L))

            # =========== Compute Loss =========

            loss = criterion(out, labels)
            running_loss += loss.data
            # ============ Backward ============
            loss.backward()
            optimizer.step()

            if args.verbose:
                print("Iteration number: ", i)
                #grid_imshow(inputs, inputs)

            # ============ Logging ============
            if i % 1000 == 999:
                if args.wandb_on:
                    wandb.log({"Finetuning Loss": running_loss / 1000,
                           "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        ''' Do Validation: After every epoch to check for overfitting '''
        if args.valid:
            classifier.eval()
            n_samples = 0.
            n_correct_top_1 = 0
            n_correct_top_k = 0

            for j, (img, target, _) in enumerate(loader_val_sup, 0):
                img = get_torch_vars(img)
                L_ = img[:, 0, :, :]  # one channel
                ab_ = img[:, 1:3, :, :]  # two channels
                target = get_torch_vars(target)
                batch_size = img.shape[0]
                n_samples += batch_size

                # ============ Forward ============
                output = classifier((ab_, L_))

                # ============ Accuracy ============
                # Top 1 accuracy
                pred_top_1 = torch.topk(output, k=1, dim=1)[1]
                n_correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).int().sum().item()

                # Top k accuracy
                top_k = 5
                pred_top_k = torch.topk(output, k=top_k, dim=1)[1]
                target_top_k = target.view(-1, 1).expand(batch_size, top_k)
                n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()

            # Accuracy
            top_1_acc = n_correct_top_1 / n_samples
            top_k_acc = n_correct_top_k / n_samples

            # Early Stopping
            if(top_1_acc < prev_top1):
                print("Early stopping triggered.")
                exit(0)
            else:
                prev_top1 = top_1_acc

            # ============ Logging ============
            if args.wandb_on:
                wandb.log({"Top-1 Accuracy": top_1_acc,
                           "Top-5 Accuracy": top_k_acc})
            print('Validation top 1 accuracy: %f' % top_1_acc)
            print('Validation top %d accuracy: %f'% (top_k, top_k_acc))

        ''' Save Trained Model '''
        print('Done Training. Saving Model...')
        torch.save(classifier.state_dict(), finetuned_weight_name)

    exit(0)

if __name__ == '__main__':
    main()
