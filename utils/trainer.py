import torch
from .progress import Bar, np
from sklearn.metrics import average_precision_score


######################################################################################################################
#                                               train loss function                                                  #
######################################################################################################################
# Define the training function for the model.
def base_train(model, train_loader, loss_calculator, optimizer, writer,
               EPOCH, DEVICE, opt):
    # Initialization of variables to track training loss and number of batches.
    train_loss = 0
    batch_num = 0

    # Train
    model.train()

    # Iterate through batches of data from the train_loader.
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        loss = loss_calculator(input=outputs, target=targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Update model parameters based on gradients.

        train_loss += loss.item()
    train_loss /= batch_num

    # Log the average training loss to tensorboard.
    writer.log_train_loss('total', train_loss, EPOCH)

    return train_loss


######################################################################################################################
#                                               valid loss function                                                  #
######################################################################################################################
# Define the validation function for the model.
def base_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # Initialization of variables to track validation loss and number of batches.
    valid_loss = 0
    batch_num = 0

    t_all = []
    o_all = []

    # Validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            p = torch.sigmoid(outputs)
            loss = loss_calculator(input=outputs, target=targets)

            valid_loss += loss

            t_all.append(targets.data.cpu().numpy())
            o_all.append(p.data.cpu().numpy())
        valid_loss /= batch_num

    t_all = np.concatenate(t_all, axis=0)
    o_all = np.concatenate(o_all, axis=0)

    # Compute the Area Under the Precision-Recall Curve (AUPRC).
    valid_auprc = average_precision_score(y_true=t_all, y_score=o_all)
    # Log the average validation loss to tensorboard.
    writer.log_valid_loss('total', valid_loss, EPOCH)
    writer.log_score('AUPRC', valid_auprc, EPOCH)

    return valid_loss, valid_auprc, t_all, o_all

