#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os

import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid, credit_iid, credit_noniid
import torch.nn.functional as F
import pandas as pd
import numpy as np


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'credit':
        dataset = pd.read_csv('/home/cc/PPFL-v2/data/credit/credit.csv')

        train_dataset = dataset[:int(len(dataset)*0.8)]
        test_dataset = dataset[int(len(dataset)*0.8):]

        # sample training data amongst users
        if args.iid:
            # Sample IID
            user_groups = credit_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = credit_iid(train_dataset, args.num_users)

    elif args.dataset == 'loans':
        dataset = pd.read_csv('/home/cc/PPFL-v2/data/loans/mortgageloans.csv')

        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]

        # sample training data amongst users
        if args.iid:
            # Sample IID
            user_groups = credit_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = credit_iid(train_dataset, args.num_users)

    elif args.dataset == 'adult_income':
        dataset = pd.read_csv('/home/cc/PPFL-v2/data/adult_income/adult_income.csv')

        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]

        # sample training data amongst users
        if args.iid:
            # Sample IID
            user_groups = credit_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = credit_iid(train_dataset, args.num_users)

    elif args.dataset == 'lfw':
        data_dir = '../data/lfw/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.ImageFolder('/home/cc/PPFL-v2/data/lfw/Images_train', transform=apply_transform)
        test_dataset = datasets.ImageFolder('/home/cc/PPFL-v2/data/lfw/Images_test', transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def private_label_obtention(args, labels):
    # Generating private attribute

    if args.dataset == 'cifar':
        # Cifar dataset doesn't have a private label. We will set as private label the image belonging to an animal '1' or not '0'
        animals = [2, 3, 4, 5, 6, 7]
        private_labels = torch.zeros(len(labels), dtype=torch.int32)
        for counter in range(len(private_labels)):
            if labels[counter] in animals:
                private_labels[counter] = 1
            else:
                private_labels[counter] = 0
    elif args.dataset == 'lfw':
        private_labels = torch.zeros(len(labels), dtype=torch.int32)
        women = [1, 4, 6, 7]
        for counter in range(len(private_labels)):
            if labels[counter] in women:
                private_labels[counter] = 0
            else:
                private_labels[counter] = 1

    return private_labels


def get_data(args, dataset):
    # For non-image data, we need to split the labels and the actual training data

    if args.dataset == 'credit':
        if args.private_attr == 0:
            #To use Gender as private attribute, use this code
            data = dataset.drop(columns=['CreditScore', 'Gender'])
            private_labels = dataset['Gender']
        elif args.private_attr == 1:
            # To use Geography as private attribute, use this code
            data = dataset.drop(columns=['CreditScore', 'Geography'])
            private_labels = dataset['Geography']

        labels = dataset['CreditScore']

    elif args.dataset == 'loans':
        if args.private_attr == 0:
            # To use Gender as private attribute, use this code
            data = dataset.drop(columns=['Affordability', 'Gender'])
            private_labels = dataset['Gender']
        elif args.private_attr == 1:
            # To use Race as private attribute, use this code
            data = dataset.drop(columns=['Affordability', 'Race'])
            private_labels = dataset['Race']

        labels = dataset['Affordability']

    elif args.dataset == 'adult_income':
        if args.private_attr == 0:
            # To use Gender as private attribute, use this code
            data = dataset.drop(columns=['Income', 'Gender'])
            private_labels = dataset['Gender']
        elif args.private_attr == 1:
            # To use Marital status as private attribute, use this code
            data = dataset.drop(columns=['Income', 'Marital_status'])
            private_labels = dataset['Marital_status']

        labels = dataset['Income']

    return data, labels, private_labels


def compute_metrics(output, real_value):

    total, correct = 0.0, 0.0

    # Compute the accuracy
    _,pred = torch.max(output, 1)
    pred_labels = pred.view(-1)
    correct += torch.sum(torch.eq(pred_labels, real_value)).item()
    total += len(real_value)
    accuracy = correct / total
    return accuracy

def jsd_loss(MI_estimator, x, z, u, x_prime):

    Ej = (-F.softplus(-MI_estimator(x, z, u))).mean()
    Em = F.softplus(MI_estimator(x_prime, z, u)).mean()
    return Ej - Em

def exp_details(args):
    print('\nExperimental details:')
    if args.dataset == 'cifar':
        print(f'    Dataset : Cifar')
    elif args.dataset == 'credit':
        print(f'    Dataset : Credit')
    elif args.dataset == 'loans':
        print(f'    Dataset : Mortgage loans')
    elif args.dataset == 'adult_income':
        print(f'    Dataset : Adult Income')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


