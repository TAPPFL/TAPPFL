#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
import update_data, update_images
#from update_images import Pretrain, LocalUpdate
from models.classifiers import Classifier_images_cifar, Classifier_images_lfw, Classifier_credit
from models.mi_model import mi_estimator_cifar, mi_estimator_lfw, mi_estimator_credit, mi_estimator_loans, mi_estimator_adult_income
from models.fe_models import FeatureExtractor_images, FeatureExtractor_credit, FeatureExtractor_loans, FeatureExtractor_adult_income
from utils import get_dataset, average_weights, exp_details

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from matplotlib import cm


if __name__ == '__main__':

    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('/home/cc/logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)


    # Definition of the models for the neural networks
    if (args.dataset == 'cifar') or (args.dataset == 'lfw'):
        fe_model = FeatureExtractor_images()
        if args.dataset == 'cifar':
            classifier = Classifier_images_cifar(args=args)
            mi_model = mi_estimator_cifar(args=args)
        elif args.dataset == 'lfw':
            classifier = Classifier_images_lfw(args=args)
            mi_model = mi_estimator_lfw(args=args)

    elif (args.dataset == 'credit') or (args.dataset == 'loans') or (args.dataset == 'adult_income'):
        if args.dataset == 'credit':
            fe_model = FeatureExtractor_credit()
            mi_model = mi_estimator_credit(args=args)
        elif args.dataset == 'loans':
            fe_model = FeatureExtractor_loans()
            mi_model = mi_estimator_loans(args=args)
        elif args.dataset == 'adult_income':
            fe_model = FeatureExtractor_adult_income()
            mi_model = mi_estimator_adult_income(args=args)
        classifier = Classifier_credit(args=args)


    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Set the models to train and send it to device.
    fe_model.to(device)
    fe_model.train()
    classifier.to(device)
    mi_model.to(device)

    # PRETRAINING STAGE: Feature Extractor and Classifier are pretrained to accurately predict the desired label
    if (args.dataset == 'cifar') or (args.dataset == 'lfw'):
        pretraining_stage = update_images.Pretrain(args=args, dataset=train_dataset, logger=logger)
    elif (args.dataset == 'credit') or (args.dataset == 'loans') or (args.dataset == 'adult_income'):
        pretraining_stage = update_data.Pretrain(args=args, dataset=train_dataset, logger=logger)
    #initialization
    fe_model, classifier = pretraining_stage.pretrain_fe(fe_model=copy.deepcopy(fe_model), classifier=copy.deepcopy(classifier))
    pretraining_stage.eval_pretrain_fe(fe_model=fe_model, classifier=classifier)

    # TRAINING STAGE: The 3-NN schema is trained on the client side using the pretrained Feature Extractor and Classifier.
    #                 Then, the weights of the Feature Extractor are sent to the server to average them and send them back
    #                 to the clients.
    global_weights = fe_model.state_dict() # Stores weights of the Feature Extractor model located on the server

    df = pd.DataFrame(index=range(50), columns=['UtilityCE', 'PrivacyCE', 'Lambda'])
    k = 0
    #for i in range(0,101,2):
        #args.tradeoff_lambda = i/100

    for epoch in tqdm(range(args.epochs)):
        local_weights_fe, local_weights_u, local_weights_p, local_acc_p, local_acc_u = [], [], [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        fe_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # Random set of m clients (St)
        n_user = 1

        for idx in idxs_users: # for each client in St do (IN PARALLEL):
            print('User {} of {}'.format(n_user, len(idxs_users)))
            if (args.dataset == 'cifar') or (args.dataset == 'lfw'):
                local_model = update_images.LocalUpdate(args=args, dataset=train_dataset,
                                                        idxs=user_groups[idx], logger=logger)
            elif (args.dataset == 'credit') or (args.dataset == 'loans') or (args.dataset == 'adult_income'):
                local_model = update_data.LocalUpdate(args=args, dataset=train_dataset,
                                                        idxs=user_groups[idx], logger=logger)

            # Train FE using the complete 3-NN setting
            w_fe = local_model.train_fe(
                fe_model=copy.deepcopy(fe_model), classifier=copy.deepcopy(classifier),
                mi_model=copy.deepcopy(mi_model), global_round=epoch+1) # Update their weights and the loss

            local_weights_fe.append(copy.deepcopy(w_fe)) # Weights are stored
            n_user+=1

        # Update global weights of FE
        global_weights = average_weights(local_weights_fe)
        fe_model.load_state_dict(global_weights)

        #Calculate avg training accuracy over all users every 5 epochs
        train_acc_privacy, train_acc_utility = [], []
        fe_model.eval()
        if (epoch+1) % 5 == 0:
            for c in range(args.num_users):
                if (args.dataset == 'cifar') or (args.dataset == 'lfw'):
                    local_model = update_images.LocalUpdate(args=args, dataset=train_dataset,
                                                            idxs=user_groups[c], logger=logger)
                elif (args.dataset == 'credit') or (args.dataset == 'loans') or (args.dataset == 'adult_income'):
                    local_model = update_data.LocalUpdate(args=args, dataset=train_dataset,
                                                            idxs=user_groups[c], logger=logger)
                acc_u_train, acc_p_train = local_model.eval_train(fe_model=fe_model, classifier=classifier, mi_model=mi_model) # Obtain training statistics

            train_acc_utility.append(acc_u_train)
            train_acc_privacy.append(acc_p_train)

            print('Average utility accuracy on training: {}% | Average privacy accuracy on training: {}%'
                  .format(round(sum(train_acc_utility)*100/len(train_acc_utility),3),
                          round(sum(train_acc_privacy)*100/len(train_acc_privacy),3)))


    # EVALUATION STAGE: Evaluate on the test set for the subset of users
    n_user=1
    test_acc_utility, test_acc_privacy, test_loss_utility, test_loss_privacy = [], [], [], []
    target_list, feature_list = [],[]
    for idx in idxs_users:
        fe_model.eval()
        if (args.dataset == 'cifar') or (args.dataset == 'lfw'):
            local_model = update_images.LocalUpdate(args=args, dataset=train_dataset,
                                                    idxs=user_groups[idx], logger=logger)
        elif (args.dataset == 'credit') or (args.dataset == 'loans') or (args.dataset == 'adult_income'):
            local_model = update_data.LocalUpdate(args=args, dataset=train_dataset,
                                                    idxs=user_groups[idx], logger=logger)

        acc_u_test, acc_p_test, loss_u_test, loss_p_test = \
            local_model.eval_test(fe_model=fe_model, classifier=classifier, mi_model=mi_model)
        test_acc_utility.append(acc_u_test)
        test_acc_privacy.append(acc_p_test)
        test_loss_utility.append(loss_u_test)
        test_loss_privacy.append(loss_p_test)
        n_user+=1

    local_model.t_sne_plot(fe_model=fe_model)

    avg_test_acc_u = sum(test_acc_utility)/len(test_acc_utility)
    avg_test_acc_p = sum(test_acc_privacy)/len(test_acc_privacy)
    var_test_acc_p = sum([((i - avg_test_acc_p) ** 2) for i in test_acc_privacy]) / len(test_acc_privacy)
    stdev_test_acc_p = var_test_acc_p**0.5

    avg_test_loss_u = sum(test_loss_utility) / len(test_loss_utility)
    avg_test_loss_p = sum(test_loss_privacy) / len(test_loss_privacy)

    print('Lambda: {}'.format(args.tradeoff_lambda))

    print('Utility accuracy on test - Mean: {}% \nPrivacy accuracy on test - Mean: {}% | Standard deviation: {}'
          .format(round(avg_test_acc_u*100,3),
                  round(avg_test_acc_p*100,3), stdev_test_acc_p))
    print('Utility CE on test - Mean: {} \nPrivacy CE on test - Mean: {}'
          .format(avg_test_loss_u,
                  avg_test_loss_p))

        #df['UtilityCE'][k] = avg_test_loss_u
        #df['PrivacyCE'][k] = avg_test_loss_p
        #df['Lambda'][k] = args.tradeoff_lambda
        #k+=1

    #df.to_csv('PPFL-v2/data/cifar/test_loss.csv', index=False)


