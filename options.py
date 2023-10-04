"""
Docstring for Options:
This class serves to provide a centralized argument parser. It defines the hyperparameters and settings
used throughout the ECG classification training process, including data processing, model architecture,
and training configurations.
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # Global settings
        parser.add_argument('--batch_size', type=int, default=24,
                            help='Size of each batch during training.')
        parser.add_argument('--nepoch', type=int, default=30,
                            help='Total number of training epochs.')
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='Type of optimizer to use during training (e.g., "adam", "sgd").')
        parser.add_argument('--lr_initial', type=float, default=1e-3,
                            help='Initial learning rate.')
        parser.add_argument("--decay_epoch", type=int, default=20,
                            help="Epoch number from which to start learning rate decay.")
        parser.add_argument('--weight_decay', type=float, default=0.1,
                            help='Weight decay factor for regularization.')
        parser.add_argument('--valid_ratio', type=float, default=0.2,
                            help='Proportion of data to be used for validation.')

        # Training settings
        parser.add_argument('--arch', type=str, default='ResU_Dense',
                            help='Model architecture to be used for training (e.g., "ResU_Dense").')
        parser.add_argument('--loss_type', type=str, default='base',
                            help='Type of loss function to be used (e.g., "base", "mse").')
        parser.add_argument('--loss_oper', type=str, default='base',
                            help='Operation type of the loss function (e.g., "add", "multiply").')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to be used for training ("cuda" for GPU, "cpu" for CPU).')

        # Network settings
        parser.add_argument('--class_num', type=int, default=26,
                            help='Number of classes for ECG classification.')
        parser.add_argument('--leads', type=int, default=12,
                            help='Number of ECG leads to be considered.')

        # Pretrained model settings
        parser.add_argument('--env', type=str, default='0901',
                            help='Name used for logging purposes.')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='Option to load weights from a pretrained model.')
        parser.add_argument('--pretrain_model_path', type=str,
                            default='./log/ResU_Dense/models/chkpt_opt.pt',
                            help='Path to the pretrained model weights.')

        # Dataset settings
        parser.add_argument('--fs', type=int, default=500,
                            help='Sampling frequency of the ECG data.')
        parser.add_argument('--samples', type=int, default=4096,
                            help='Number of samples to consider from the ECG data.')
        parser.add_argument('--dirs_for_train', type=str,
                            default='../Dataset/',
                            help='Directory path where ECG dataset for training and validation is located.')

        return parser
