from __future__ import print_function
import argparse
import queue
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, q, error_q, optimizer, epoch, train_len, train_len2, log_interval, dry_run):
    model.train()
    # Run the training until we are notified to stop by the data loading thread.
    while True:
        # First, check to see if the other thread encountered an error. If so,
        # exit the program to prevent hang. Otherwise, get the "product" from the
        # queue, and check to see if it is the "done" notification. If so, break the
        # loop. If not, run the training for this batch. If this thread encounters
        # an error, send an "Error" message in the queue to notify the other thread,
        # in order to prevent hang, and then raise the exception.
        try:
            if error_q.qsize() > 0:
                err = error_q.get()
                if err == 'Error': return

            product = q.get()
            if product == "done": break

            batch_idx, data, target = product
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), train_len,
                           100. * batch_idx / train_len2, loss.item()))
                if dry_run:
                    break
        except:
            print("An exception occurred")
            error_q.put("Error")
            raise


def test(model, device, q, error_q):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # Run the testing until we are notified to stop by the data loading thread.
        while True:
            # First, check to see if the other thread encountered an error. If so,
            # exit the program to prevent hang. Otherwise, get the "product" from the
            # queue, and check to see if it is the "done" notification. If so, break the
            # loop. If not, run the training for this batch. If this thread encounters
            # an error, send an "Error" message in the queue to notify the other thread,
            # in order to prevent hang, and then raise the exception.
            try:
                if error_q.qsize() > 0:
                    err = error_q.get()
                    if err == 'Error': return

                product = q.get()
                if product == "done": break

                data, target = product
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            except:
                print("An exception occurred")
                error_q.put("Error")
                raise

    return test_loss, correct

# A function to handle all of the data loading and management
def io(args, q, error_q):
    # First, we load the data and define some variables
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the device in the queue; the main program needs it for the model.
    # Also put various other values necessary for display purposes into the queue.
    q.put(device)
    q.put(args.epochs)
    q.put(len(test_loader.dataset))
    q.put(len(train_loader.dataset))
    q.put(len(train_loader))
    q.put(args.log_interval)
    q.put(args.dry_run)
    # Loop through each batch in each epoch. First, check to see if the other
    # thread encountered an error, and if they did, exit the program to prevent
    # hang. If not, put the training batch index, data, and labels into the
    # queue. Once all batches have been sent, send a "done" message to let the
    # other thread know to stop and move on to the next epoch. The same is done
    # for the test set as well. If this thread encounters an error, put an "Error"
    # message into the queue to notify the other thread, then raise the error.
    try:
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                if error_q.qsize() > 0:
                    if error_q.get() == 'Error': return
                q.put((batch_idx, data, target))
            q.put("done")

            for data, target in test_loader:
                if error_q.qsize() > 0:
                    if error_q.get() == 'Error': return
                q.put((data, target))
            q.put("done")
    except:
        print("An exception occurred")
        error_q.put("Error")
        raise


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Initialize an error queue, to allow the different threads to notify each other of
    # errors, in order to prevent the program from hanging.
    error_q = queue.Queue()
    # Initialize a queue, in which to send the data, once it is loaded, to the threads that
    # will utilize that data.
    q = queue.Queue()
    # Initialize a separate thread to run the data loading.
    io_t = threading.Thread(target=io, args=(args, q, error_q))

    # Start the data loading thread, and get the device, the total number of epochs, as well as
    # various other variables necessary for display purposes.
    io_t.start()
    device = q.get()
    epochs = q.get()
    test_len = q.get()
    train_len = q.get()
    train_len2 = q.get()
    log_interval = q.get()
    dry_run = q.get()

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Loop through each epoch. For every loop, check to see if the other thread had an error,
    # and if not, run the training function, the testing function, and print the loss and
    # accuracy. If this thread has an error, put an "Error" message in the error queue, and
    # raise the exception.
    for epoch in range(1, epochs + 1):
        print("Epoch: ", epoch)
        try:
            if error_q.qsize() > 0:
                err = error_q.get()
                if err == 'Error': break
            train(model, device, q, error_q, optimizer, epoch, train_len, train_len2, log_interval, dry_run)
            test_loss, correct = test(model, device, q, error_q)
            scheduler.step()
            test_loss /= test_len
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, test_len,
                100. * correct / test_len))
        except:
            print("An exception occurred")
            error_q.put("Error")
            raise
    io_t.join()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
