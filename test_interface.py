"""
Test interface for ECG classification!
You can just run this file.
"""
import os
import argparse
import torch
import options
import datetime
import random
import utils
import numpy as np
from dataloader import create_dataloader_for_test
import sklearn
######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)

######################################################################################################################
#                                    Set a model (check point) and a log folder                                      #
######################################################################################################################
dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
print(dir_name)

log_dir = os.path.join(dir_name, 'log', opt.arch + '_' + opt.env)

utils.mkdir(log_dir)
print("Now time is : ", datetime.datetime.now().isoformat())
tboard_dir = './log/{}_{}/logs'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'logs')
model_dir = './log/{}_{}/models'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'models')
utils.mkdir(model_dir)  # make a dir if there is no dir (given path)
utils.mkdir(tboard_dir)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)

# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# define model
model = utils.get_arch(opt)

total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# load the params
print('Load the pretrained model...')
chkpt = torch.load(opt.pretrain_model_path)
model.load_state_dict(chkpt['model'])

model = model.to(DEVICE)

######################################################################################################################
######################################################################################################################
#                                             Main program - test                                                    #
######################################################################################################################
######################################################################################################################
test_loader = create_dataloader_for_test(opt, test_dataset_addr=opt.test_dataset_addr)
test_log_fp = open(model_dir + '/test_log.txt', 'a')

t_all = []
o_all = []
b_all = []

model.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        q = torch.sigmoid(outputs)

        t_all.append(targets.data.cpu().numpy())
        o_all.append(q.data.cpu().numpy())

t_all = np.concatenate(t_all, axis=0)
o_all = np.concatenate(o_all, axis=0)

threshold = [0.5 for _ in range(opt.class_num)]
print('Threshold: ', threshold)

for idx in range(len(o_all)):
    labels = utils.get_binary_outputs_np(o_all[idx], threshold)
    b_all.append(labels)

b_all = np.array(b_all)

######################################################################################################################
#                                                   Get scores                                                       #
######################################################################################################################
num_classes = opt.class_num

auroc, auprc, auroc_classes, auprc_classes = utils.compute_auc(t_all, o_all)
accuracy = utils.compute_accuracy(t_all, b_all)
f_measure, f1score_classes, precision, precision_classes, recall, recall_classes, specificity, specificity_classes = utils.compute_metrics(t_all, b_all)

np.save('./results/{}_F1_score.npy'.format(opt.exp_name), f1score_classes)
np.save('./results/{}_AUPRC.npy'.format(opt.exp_name), auprc_classes)
np.save('./results/{}_Precision.npy'.format(opt.exp_name), precision_classes)
np.save('./results/{}_Specificity.npy'.format(opt.exp_name), specificity_classes)
np.save('./results/{}_Recall.npy'.format(opt.exp_name), recall_classes)

kappa = utils.compute_kappa(t_all, b_all, num_classes=num_classes)
hamming_loss = sklearn.metrics.hamming_loss(t_all, b_all)
print("####################################################################")
print(opt.pretrain_model_path)
print("Accuracy : {:.4}".format(accuracy))
print("Kappa : {:.4}".format(kappa))
print("AUROC : {:.4}".format(auroc))
print("precision : {:.4}".format(precision))
print("recall : {:.4}".format(recall))
print("specificity : {:.4}".format(specificity))
print("AUPRC : {:.4}".format(auprc))
print("F-Measure : {:.4}".format(f_measure))
print("Hamming loss : {:.4}".format(hamming_loss))
print("####################################################################")

print('System has been finished.')
