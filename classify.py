from utils import *
from load_data import *
import os
import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision

import wandb

def main():
    wandb.init()

    # Parse Args
    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder")
    parser.add_argument("--valid", '--do_validation',type=bool, default=False,
                        help="Perform validation only.", metavar='v')
    parser.add_argument("--perc_noise", '-percentage_of_noise', type=float, default=0.05,
                        help="Percentage of noise to add.", metavar='p')
    parser.add_argument("--corr_type", '-type_of_noise', type=str, default="mask",
                        help="Percentage of noise to add.", metavar='c')
    parser.add_argument("--verbose", '-verbose_mode', type=bool, default=False,
                        help="Show images as you feed them in, show reconstructions as they come out.", metavar='b')
    parser.add_argument("--wandb", '-name_of_wandb_proj', type=str, default="le-project",
                        help="Name of WAND Project.", metavar='w')
    args = parser.parse_args()

    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_weight_name = os.path.join(file_path, "weights/%s/ae_%s.pkl" % (args.corr_type, str(args.perc_noise)))
    finetuned_weight_name = os.path.join(file_path,"weights/%s/ae_finetuned_%s.pkl" % (args.corr_type, str(args.perc_noise)))

    if not os.path.exists(os.path.join(file_path, "weights")) or os.path.join(file_path, "weights/%s/ae_%s.pkl" % (args.corr_type, str(args.perc_noise))):
        raise Exception('Your pretrained weights folder is missing')
        exit(1)


    wandb.config.update(args)


    # Create model
    classifier = create_model("classify")

    # Load pretrained weights
    classifier.ae.load_state_dict(torch.load(pretrained_weight_name))

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_image_loader("../ssl_data_96", 32)

    # Define an optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    wandb.watch(classifier)

    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader_sup, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            out = classifier(inputs)
            loss = criterion(out, labels)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 1000 == 999:
                wandb.log({"Finetuning Loss": running_loss / 1000,
                           "Epoch" : epoch + 1,
                           "Iteration" : i + 1,
                           })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        ''' Save Trained Model '''
        print('Done Training. Saving Model...')
        if not os.path.exists(os.path.join(file_path,'weights/%s' % args.corr_type)):
            os.mkdir(os.path.join(file_path,'weights/%s' % args.corr_type))
        torch.save(classifier.state_dict(), finetuned_weight_name)

        ''' Do Validation: After every epoch to check for overfitting '''
        if args.valid:

            val_loss = 0

            for j, (img, label) in enumerate(loader_val_sup, 0):
                inputs = get_torch_vars(inputs)

                # ============ Forward ============
                decoded_val, out_val = classifier(img)
                next_loss = criterion(out_val, label)
                val_loss += next_loss
                # ============ Verbose ============
                if args.verbose:
                    decoded_img = decoded_val[1]
                    imshow(torchvision.utils.make_grid(img))
                    imshow(torchvision.utils.make_grid(decoded_img.data))

            # ============ Logging ============
            wandb.log({"Validation Loss": val_loss})
            print('[%d, %5d] Validation loss: %.3f' % (epoch + 1, j + 1, val_loss))

    exit(0)

if __name__ == '__main__':
    main()
