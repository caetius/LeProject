# Torch
import torch.nn as nn
import torch.optim as optim
from load_data import *

# This project
from denoise import corrupt_input
from utils import *

# Torchvision
import torchvision

# OS
import os
import argparse

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
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--perc_noise", action="store_true", default=0.1,
                        help="Percentage of noise to add.")
    parser.add_argument("--corr_type", action="store_true", default="mask",
                        help="Percentage of noise to add.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Show images as you feed them in, show reconstructions as they come out.")
    args = parser.parse_args()

    # Create model
    ae = create_model("pretrain")

    ''' Load data '''
    loader_sup, loader_unsup, loader_val_sup = nyu_image_loader("../ssl_data_96", 32)

    ''' Do Validation '''
    if args.valid:
        print("Loading checkpoint...")
        ae.load_state_dict(torch.load("./weights/ae.pkl"))
        dataiter = iter(loader_val_sup)
        images, labels = dataiter.next()
        #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = ae(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae.parameters())

    for epoch in range(10):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(loader_sup, 0):
            inputs = get_torch_vars(inputs)
            print(inputs.shape)
            noised = corrupt_input(args.corr_type, inputs, args.perc_noise)
            print("Iteration ", i)
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
                imshow(outputs[0])

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    ''' Save Trained Model '''
    print('Done Training. Saving Model...')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    torch.save(ae.state_dict(), "./weights/ae.pkl")


if __name__ == '__main__':
    main()
