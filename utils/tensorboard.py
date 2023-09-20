from tensorboardX import SummaryWriter


# Define a custom writer class that extends SummaryWriter to log training/validation metrics.
class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    # Method to log training loss.
    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    # Method to log validation loss.
    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)

    # Method to log other performance metrics (e.g., accuracy, F1-score).
    def log_score(self, metrics_name, metrics, step):
        # Add a scalar value to the writer with the given metric name.
        self.add_scalar(metrics_name, metrics, step)



