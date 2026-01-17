from boitenoire import*

device = "cuda:0" if torch.cuda.is_available() else "cpu" # GPU use

#Importation & preprocessing
# MNIST Importation
train_data, test_data = load_mnist()

### Cleared MNIST Importation
traindata_cleared, testdata_cleared = load_mnist(True)

# Différence de qualité pour une image donnée
plt.subplot(1, 2, 1)
plt.imshow(train_data[4][0].view(28, 28), cmap = "gray")
plt.title("A gauche l'image originale et à droite l'image améliorée :", loc = "left")
plt.subplot(1, 2, 2)
plt.imshow(traindata_cleared[4][0].view(28, 28), cmap = "gray")
plt.show()

# Réduction de la taille du jeu de données
traindata_reduced = reduce_mnist(traindata_cleared)

# Affichage du jeu de données réduit

fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize = (10, 10))

for i in range(10) :
    for j in range(10) :
        img, label = traindata_reduced[i*10 + j]
        ax[i, j].imshow(img.squeeze(), cmap = 'gray')
        ax[i, j].axis('off')
        ax[i, j].set_title('Label: {}'.format(label))

plt.tight_layout()
plt.show()

# Afficher l'image originale et ses rotations pour une image donnée
rotated_dataset = rotate_mnist(traindata_reduced)
for i in range(0,100,10):
  display_rotated_images(rotated_dataset, index=i)

class Rotnet(nn.Module) :
    """Rotnet for digit recognition task"""

    def __init__(self) :
        super().__init__()

        # Premier bloc de convolution
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Second bloc de convolution
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = "same")
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same")
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = "same")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Couches entièrement connectées
        self.fc1 = nn.Linear(in_features = 3136, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 4)

    def forward(self, x) :

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN(nn.Module) :
    """CNN for classification purpose"""

    def __init__(self) :
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = "same")
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(in_features = 6272, out_features = 1024)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 10)

    def forward(self, x) :

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

get_model_memory(CNN())

get_model_memory(Rotnet())

# MNIST dataset
traindata, testdata = load_mnist(True)

### Dataset and préparation du modèle
# Jeux de données avec rotation des images
traindata_rotated = rotate_mnist(traindata)
testdata_rotated = rotate_mnist(testdata)

# Séparation en set d'entrainement et de tes
traindata_rotated, validata_rotated = split_train_valid(traindata_rotated, 0.2, random_state=None) #80% pour l'entrainement et 20% pour la validation

# Taille des jeux de données
print("Size of the train dataset :", len(traindata_rotated), ",", traindata_rotated[0][0].shape,
      "\nSize of the validation dataset :", len(validata_rotated), ",", validata_rotated[0][0].shape,
      "\nSize of the test dataset :", len(testdata_rotated), ",", testdata_rotated[0][0].shape, "\n")

# Chargement des données dans un dataloader
trainloader = load_data(traindata_rotated, 256, True, 0)
validloader = load_data(validata_rotated, 256, False, 0)
testloader = load_data(testdata_rotated, 256, False, 0)

### Entrainement du modèle
rotnet = Rotnet().to(device)

# Définition des paramètres
output_fn = nn.Softmax(dim = 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rotnet.parameters(), lr = 0.01)

# Entrainement du ROTNET
epochs = 20
args_train = {"train_loader" : trainloader,
              "val_loader" : validloader,
              "model" : rotnet,
              "output_fn" : output_fn,
              "epochs" : epochs,
              "optimizer" : optimizer,
              "criterion" : criterion,
              "device" : device}

rotnet, (loss_train, acc_train, loss_valid, acc_valid) = train_model(**args_train)

plot_accuracy(epochs, loss_train, loss_valid, acc_train, acc_valid)

evaluate_model(rotnet, testloader, device, 4)

import os
os.makedirs("save_models", exist_ok=True) # Créé la direction si elle n'existe pas

# Sauvegarde du modèle
save_model(rotnet, "save_models/rotnet.pth")

### Fine tuning du modèle pré entrainé
rotnet = Rotnet().to(device)
rotnet.load_state_dict(torch.load("save_models/rotnet.pth", map_location = device))

# Paramètres du modèle
params = rotnet.state_dict()
params.keys()

# Figer les paramètres sauf ceux de la dernière couche

for name, param in rotnet.named_parameters() :
    if param.requires_grad and 'conv'  in name :
        param.requires_grad = False

# Vérification des paramètres figés

for name, param in rotnet.named_parameters() :
    print(name, param.requires_grad)

### Mise à jour des dimensions de la dernière couche

# Récupérer le nombre d'entités en entrée dans la dernière couche entièrement connectée
number_features_last_layer = rotnet.fc2.in_features

# Réinitialiser la dernière couche avec le nb correct de sorties
rotnet.fc2 = nn.Linear(number_features_last_layer, 10)

# Récapitulatif du modèle
rotnet = rotnet.to(device)
get_model_memory(rotnet)

### Dataset et préparation du modèle

# Réduction du nombre d'images
traindata_reduced = reduce_mnist(traindata)
traindata_reduced, validata_reduced = split_train_valid(traindata_reduced, 0.2, False)

# Taille des données
print("Size of the train dataset :", len(traindata_reduced), ",", traindata_reduced[0][0].shape,
      "\nSize of the validation dataset :", len(validata_reduced), ",", validata_reduced[0][0].shape,
      "\nSize of the test dataset :", len(testdata), ",", testdata[0][0].shape, "\n")

# Chargement des données
trainloader = load_data(traindata_reduced, 80, True, 0)
validloader = load_data(validata_reduced, 20, False, 0)
testloader = load_data(testdata, 128, False, 0)

### Entrainement du modèle

# Paramètres
output_fn = nn.Softmax(dim = 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, rotnet.parameters()), lr = 0.01)

# Entrainement
epochs = 20
args_train = {"train_loader" : trainloader,
              "val_loader" : validloader,
              "model" : rotnet,
              "output_fn" : output_fn,
              "epochs" : epochs,
              "optimizer" : optimizer,
              "criterion" : criterion,
              "device" : device}

rotnet, (loss_train, acc_train, loss_valid, acc_valid) = train_model(**args_train)

plot_accuracy(epochs, loss_train, loss_valid, acc_train, acc_valid)

evaluate_model(rotnet, testloader, device, 10)

### Entrainement du modèle CNN
cnn= CNN().to(device)

# Paramètres
output_fn = nn.Softmax(dim = 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

# Entrainement pour le CNN
epochs = 20
args_train = {"train_loader" : trainloader,
              "val_loader" : validloader,
              "model" : cnn,
              "output_fn" : output_fn,
              "epochs" : epochs,
              "optimizer" : optimizer,
              "criterion" : criterion,
              "device" : device}

cnn, (loss_train, acc_train, loss_valid, acc_valid) = train_model(**args_train)

plot_accuracy(epochs, loss_train, loss_valid, acc_train, acc_valid)

evaluate_model(cnn, testloader, device, 10)