from utils import *
from load_data import *
import os
import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision

import wandb

def main():

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
                        help="Name of WAND Project.", metavar='w1')
    parser.add_argument("--wandb_on", '-is_wand_on', type=bool, default=False,
                        help="Name of WAND Project.", metavar='w2')
    args = parser.parse_args()

    ''' IMPORTANT: Name the weights such that there's no naming conflict between runs.'''
    file_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_weight_name = os.path.join(file_path, "weights/%s/ae_%s.pkl" % (args.corr_type, str(args.perc_noise)))
    finetuned_weight_name = os.path.join(file_path,"weights/%s/ae_finetuned_%s.pkl" % (args.corr_type, str(args.perc_noise)))

    if not (os.path.exists(os.path.join(file_path, "weights")) or os.path.join(file_path, "weights/%s/ae_%s.pkl" % (args.corr_type, str(args.perc_noise)))):
        raise Exception('Your pretrained weights folder is missing')
        exit(1)

    if args.wandb_on:
        wandb.init(project=args.wandb)
        wandb.config.update(args)


    # Create model
    classifier = create_model("classify", ckpt=pretrained_weight_name)

    ''' Load data '''
    loader_sup, loader_val_sup, loader_unsup = nyu_image_loader("../ssl_data_96", 32)

    # Define an optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    if args.wandb_on:
        wandb.watch(classifier)

    for epoch in range(20):
        running_loss = 0.0

        classifier.train()

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
                if args.wandb_on:
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
            classifier.eval()
            val_loss = 0
            n_samples = 0.
            n_correct_top_1 = 0
            n_correct_top_k = 0

            for j, (img, target) in enumerate(loader_val_sup, 0):
                inputs = get_torch_vars(inputs)
                batch_size = img.size(0)
                n_samples += batch_size

                # ============ Forward ============
                output = classifier(img)
                next_loss = criterion(output, label)
                val_loss += next_loss

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

            # ============ Logging ============
            split = 'Validation'
            if args.wandb_on:
                wandb.log({"Validation Loss": val_loss,
                           "Top-1 Accuracy": top_1_acc,
                           "Top-5 Accuracy": top_5_acc})
            print('[%d, %5d] Validation loss: %.3f' % (epoch + 1, j + 1, val_loss))
            print(f'{split} top 1 accuracy: {top_1_acc:.4f}')
            print(f'{split} top {top_k} accuracy: {top_k_acc:.4f}')

    exit(0)

if __name__ == '__main__':
    main()
