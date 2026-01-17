import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Evaluation du modèle
def get_accuracy(y_true, y_pred) :
  """
  y_true: valeurs des vrais labels
  y_pred: valeurs des labels prédits
  Retourne l'accuracy du modèle
  """

  return int(np.sum(np.equal(y_true, y_pred))) / y_true.shape[0]


# Architecture et mémoire
def get_model_memory(model) :
  """Retourne l'allocation de mémoire, le nombre de paramètres et
  l'achitecture du modèle"""

  # Architecture du modèle
  print(model)
  print("Model memory allocation : {:.2e}".format(torch.cuda.memory_reserved(0)
  - torch.cuda.memory_allocated(0)))

  # Paramètres
  total_params = sum(p.numel() for p in model.parameters())
  print(f"\nNb total de paramètres : {total_params}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Nb total de paramètres d'entrainement : {total_trainable_params}")

# Importation MNIST
def load_mnist(cleared = False) :
    """Importer les données MNIST
    Si cleared = True, alors l'image est en noir et blanc"""

    if cleared :
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307, ), (0.3081, )),
                                          lambda x: x > 0,
                                          lambda x: x.float(), ])
    else :
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, ), (0.5, ))])

    # Importation des données MNIST
    train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transformer) # Train set
    test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transformer) # Test set

    return train_data, test_data


# Sélection des 100 données labélisées
def reduce_mnist(data, nb_label = 10) :
    """Retourne un sous-ensemble aléatoire de 10 images par label"""

    reduced_data = []

    for i in range(10) :
        label_indices = torch.where(data.targets == i)[0]
        subset_indices = random.sample(label_indices.tolist(), nb_label)
        reduced_data.append(Subset(data, subset_indices))

    return ConcatDataset(reduced_data)


# Rotation des images à 90°, 180° et 270°
def rotate_mnist(data) :
    """Ajout de nouvelles images par rotation de 90°, 180° et 270° de
    l'image originales"""

    rotated_data = []
    rotated_labels = []

    for i in range(len(data)) :
        img, _ = data[i]

        rotated_data.append(img)
        rotated_labels.append(0)

        # 3 nouvelles images par rotation
        for angle in [90, 180, 270] :
            img_rotated = rotate(img, angle)
            rotated_data.append(img_rotated)

        # Mise à jour de la liste des lables pour la rotation
        # 1:rotation à 90°, 2: rotation à 180°, 3: rotation à  270°
        rotated_labels.extend([1, 2, 3])

    # Conversion en tenseur
    rotated_data = torch.stack(rotated_data)
    rotated_labels = torch.tensor(rotated_labels)

    return TensorDataset(rotated_data, rotated_labels)

def load_data(data, batch_size, shuffle = False, num_workers = 2) :
    """Charge le dataset dans un dataloader"""

    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


def train_model(train_loader, val_loader, model = None, output_fn = None,
                epochs:int = None, optimizer = None, criterion = None,
                device = None) :
    """Entraine le modèle pytorch et calcule l'accuracy et la loss pour chaque epoch"""

    loss_valid, acc_valid = [], []
    loss_train, acc_train = [], []

    for epoch in tqdm(range(epochs)) :

        # Entrainement
        model.train()
        running_loss = 0.0
        for _, batch in enumerate(train_loader) :

            # Entrainement sur le GPU
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialisation du gradient à 0
            optimizer.zero_grad()

            # Forward, backward et optimizer
            out = model(x = inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        # Calcul de la  loss et de l'accuracy sur l'ensemble de validation après chaque epoch
        model.eval()
        with torch.no_grad() : # Pas besoin de calculer le gradient pour l'évaluation
            idx = 0

            for batch in val_loader :
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if idx==0 :
                    t_out = model(x = inputs)
                    t_loss = criterion(t_out, labels).view(1).item()
                    t_out = output_fn(t_out).detach().cpu().numpy()
                    t_out = t_out.argmax(axis = 1)
                    ground_truth = labels.detach().cpu().numpy()

                else :
                    out = model(x = inputs)
                    t_loss = np.hstack((t_loss, criterion(out, labels).item()))
                    t_out = np.hstack((t_out, output_fn(out).argmax(axis = 1).detach().cpu().numpy()))
                    ground_truth = np.hstack((ground_truth, labels.detach().cpu().numpy()))
                idx += 1

            acc_valid.append(get_accuracy(ground_truth, t_out))
            loss_valid.append(np.mean(t_loss))

        # Calcul de la loss et de l'accuracy sur le training set après chaque epoch
        with torch.no_grad() :
            idx = 0

            for batch in train_loader :
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if idx==0 :
                    t_out = model(x = inputs)
                    t_loss = criterion(t_out, labels).view(1).item()
                    t_out = output_fn(t_out).detach().cpu().numpy()
                    t_out = t_out.argmax(axis = 1)
                    ground_truth = labels.detach().cpu().numpy()

                else :
                    out = model(x = inputs)
                    t_loss = np.hstack((t_loss, criterion(out, labels).item()))
                    t_out = np.hstack((t_out, output_fn(out).argmax(axis = 1).detach().cpu().numpy()))
                    ground_truth = np.hstack((ground_truth, labels.detach().cpu().numpy()))
                idx += 1

        acc_train.append(get_accuracy(ground_truth, t_out))
        loss_train.append(np.mean(t_loss))

        print("| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} "\
        "| Val: Loss {:.4f} Accuracy : {:.4f}\n".format(epoch + 1, epochs, loss_train[epoch], acc_train[epoch], loss_valid[epoch], acc_valid[epoch]))

    return model, (loss_train, acc_train, loss_valid, acc_valid)


def split_train_valid(data, valid_size, random_state = None, is_image = True) :
    """Séparation du jeu de données en set d'entrainemente et de validation"""

    if is_image :

        labels = []
        for i in range(len(data)) :
            _, label = data[i]
            labels.append(label)

        # Séparation des indices pour l'entrainement et la validation si on a une image
        train_indices, val_indices = train_test_split(list(range(len(labels))), test_size = valid_size, stratify = labels, random_state = random_state)

    else :
        train_indices, val_indices = train_test_split(list(range(len(data[:][1]))), test_size = valid_size, stratify = data[:][1], random_state = random_state)

    return Subset(data, train_indices), Subset(data, val_indices)


def plot_accuracy(epochs, loss_train, loss_valid, acc_train, acc_valid) :
    """Trace la courbe de l'accuracy et de la loss"""

    fig = plt.figure(figsize = (16, 8))

    def plot_metric(epochs, metric_train, metric_valid, metric_name) :
        """Trace les métriques pour le set d'entrainement et de validation"""

        plt.plot(range(1, epochs + 1), metric_train, label = f"Training {metric_name.lower()}")
        plt.plot(range(1, epochs + 1), metric_valid, label = f"Validation {metric_name.lower()}")
        plt.xlabel("epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} functions")
        plt.legend()

    # Tracer la courbe de la loss
    ax = fig.add_subplot(121)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)
    plot_metric(epochs, loss_train, loss_valid, "Loss")

    # Tracer la courbe de l'accuracy
    ax = fig.add_subplot(122)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)
    plot_metric(epochs, acc_train, acc_valid, "Accuracy")


def evaluate_model(model, test_loader, device, num_classes = 10) :
    """Evaluation du modèle sur le set de test"""

    y_true, y_pred = [], []
    with torch.no_grad() :

        for inputs, labels in test_loader :
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Affiche le rapport de classification
    target_names = [f"Class {str(i)}" for i in range(num_classes)]
    print("Classification report :")
    print(classification_report(y_true, y_pred, target_names = target_names))

    # Affiche la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis = 1)[:, np.newaxis]
    plt.subplots(figsize = (num_classes,num_classes))
    sns.heatmap(cm, annot = True, fmt = ".2f", cmap = "plasma", square = True,
                xticklabels = target_names, yticklabels = target_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")
    plt.show()


def save_model(model, path) :
    """sauvegarde le modèle dans un fichier.pth"""

    return torch.save(model.state_dict(), path)

