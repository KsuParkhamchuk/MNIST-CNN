# TODO check the normal distribution instead of kaiming 
# TODO try training on Cpu
# TODO measure memory usage and gradient stability

import torch
from data import get_mnist_dataloader
from model import MNISTCNN
from torch import nn, optim
from timeit import default_timer as timer
from wandb_config import init_wandb, log_metrics, finish_run
import wandb

def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        # filling the weights using kaiming normal distribution, fan_in mode based on inputs size
        nn.init.kaiming_normal_(module.weight, mode = 'fan_in', nonlinearity='relu')

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_model(epochs, batch_size, learning_rate):
    # init new run to start tracking
    init_wandb()

    # Device configuration 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # Train and validation dataloader
    train_dataloader = get_mnist_dataloader('train', batch_size=batch_size)
    test_dataloader = get_mnist_dataloader('test', batch_size=batch_size)

    # Model, loss, optimizer

    # model iniatialization
    model = MNISTCNN() 
    # move model to GPU
    model.to(device)
    # default weights initialization
    model.apply(init_weights) 
    # LogSoftmax + Log Likelihood Loss (difference between preticted and actual)
    criterion = nn.CrossEntropyLoss() 
    # updates model params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    def start_training_loop(epoch):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            # clear gradients
            optimizer.zero_grad() 

            # compute prediction (logits)
            outputs = model(images) 

            # compute loss
            loss = criterion(outputs, labels) 

            # compute gradients
            # Pytorch starts from the end of computational graph, computes gradient of each operation, accumulates gradient through the chain rule
            loss.backward()

            # update parameters using the gradients
            optimizer.step()  

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                avr_loss = running_loss / 100
                print(f'Epoch N: {epoch}')
                print(f'Average loss over 100 batches = {avr_loss}')
                running_loss = 0.0
            
            log_metrics({"running loss": running_loss})


    def start_validation_loop():
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0

        # Pytorch by default stores the computational graph and tracks operations for potential backprop
        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = model(images) # tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
                loss = criterion(outputs, labels)
                val_loss += loss

                _, predicted = torch.max(outputs.data, 1) # ignore the values, get predictions (e.g. [2,0])
                total += labels.size(0) # batch size
                # predicted == labels gives e.g. [True, True, False]
                # sum counts true values => e.g tensor(3)
                # item converts to Python number
                correct += (predicted == labels).sum().item() 
               
        
        val_loss = val_loss /len(test_dataloader)
        accuracy = correct/total * 100
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy}%')
        log_metrics({"accuracy: ": accuracy, "validation loss:": val_loss})



    start = timer()
    # train num of epochs, forward pass, backward pass
    for epoch in range(epochs):
        epoch_start = timer()
        start_training_loop(epoch)
        start_validation_loop()
        epoch_end = timer()
        log_metrics({"epoch_time:": epoch_end - epoch_start, "total_training_time": timer() - start,})
        
    wandb.run.summary['parameters:'] = count_parameters(model)
    finish_run()

    #save the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs
    }

    torch.save(checkpoint, 'checkpoint.pth')