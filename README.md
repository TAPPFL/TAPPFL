# Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks

Implementation of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).
Updates done to implement a Privacy Preserving network.

For better understanding the code and the formulation check also these papers:


## Running the experiments

Federated experiment involves training a global model using many local models.

* Before running these experiments, it is necessary to install the following libraries:
```
 sudo pip3 install numpy
 sudo pip3 install pandas
 sudo pip3 install tqdm
 sudo pip3 install torch
 sudo pip3 install tensorboardX
 sudo pip3 install sklearn
 sudo pip3 install torchvision
```
EXPERIMENTS ON CIFAR-10
* To run the federated experiment with CIFAR on CNN (IID):
```
sudo python3 src/federated_main.py  --dataset=cifar --gpu=0 --iid=1 --num_classes=10
```
* To run the same experiment under non-IID condition:
```
sudo python3 src/federated_main.py  --dataset=cifar --gpu=0 --iid=0 --num_classes=10
```

EXPERIMENTS ON CREDIT
For this dataset, two different private attributes can be defined: Gender (--private_attr=0) 
and Geography (--private_attr=1). For Geography, add the option --num_classes_priv=3 
to the lines below.
* To run the federated experiment with credit dataset (IID):
```
sudo python3 src/federated_main.py  --dataset=credit --gpu=0 --iid=1
```
* To run the same experiment under non-IID condition:
```
sudo python3 src/federated_main.py  --dataset=credit --gpu=0 --iid=0
```

EXPERIMENTS ON LOANS
For this dataset, two different private attributes can be defined: Gender (--private_attr=0)
and Race (--private_attr=1).
* To run the federated experiment with credit dataset (IID):
```
sudo python3 src/federated_main.py  --dataset=loans --gpu=0 --iid=1 --num_classes=3
```
* To run the same experiment under non-IID condition:
```
sudo python3 src/federated_main.py  --dataset=loans --gpu=0 --iid=0 --num_classes=3
```

EXPERIMENTS ON ADULT INCOME
For this dataset, two different private attributes can be defined: Gender (--private_attr=0)
and Marital status (--private_attr=1). For Marital status, add the option --num_classes_priv=7
to the lines below.
* To run the federated experiment with credit dataset (IID):
```
sudo python3 src/federated_main.py  --dataset=adult_income --gpu=0 --iid=1
```
* To run the same experiment under non-IID condition:
```
sudo python3 src/federated_main.py  --dataset=adult_income --gpu=0 --iid=0
```

