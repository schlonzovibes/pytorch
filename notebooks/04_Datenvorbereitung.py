from torchvision.datasets import CIFAR10

train_data = CIFAR10(root="./train/", train=True, download=True)

print("-------------------------")
print("Printing Training Data Infos")
print(train_data)
print(len(train_data))
# Access the raw data  NumPy Array
print(train_data.data.shape)
print(train_data.classes)
print(train_data.class_to_idx)

print("-------------------------")
print("Printing Element [0]")
# Access first element directly
x = train_data[0]
print(x)
print(type(x))


# Python Tuple unpacking - create and set both variables at once
data, label = x

# Same as
# image = x[0]
# image = train_data[0][0]
print("-------------------------")
print("Printing Element [0] Data and Label")
print(type(data))
print(data)
print(type(label))
print(label)
print(train_data.classes[label])


# Use same path but other training Flag
# Downloaded file contains 5 training batches and 1 test batch
# Flag decides what parts of the archive will be used
test_data = CIFAR10(root="./train/", train=False, download=True)

print("-------------------------")
print("Printing Test Data Infos")
print(test_data)
print(len(test_data))
print(test_data.data.shape)


from torchvision import transforms

# Save global mean and std in variable for easier reuse
normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.210)
)


# Compose a transform for augmenting the training data
# Will be applied lazy: when an element is processed or accsessed
train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

train_data.transforms = train_transforms

# Compose and apply transforms for test data
test_transforms = transforms.Compose([transforms.ToTensor(), normalize])
test_data.transforms = test_transforms

import torch

# Dataloader is Iterable but itself does not follow any position inside the dataset
# Iter creates an Iterator out of it - that knows its position inside the dataset
# Next fetches the next batch through the Dataloader and sets the internal counter up
trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

data_batch, labels_batch = next(iter(trainloader))
print(data_batch.size)
print(labels_batch.size())

testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
