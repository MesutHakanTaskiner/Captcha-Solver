import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

image_folder = "dataset"

image_files = os.listdir("dataset")

train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)

print(f"Training files: {len(train_files)}, Test files: {len(test_files)}")


class CaptchaDataset(Dataset):
    def __init__(self, image_folder, image_files, transform=None):
        self.image_folder = image_folder
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 64))  
        label = self.image_files[idx].split('.')[0]  
    
        label_tensor = torch.zeros(5, 36)  
        for i, char in enumerate(label):
            if char.isdigit():
                idx = int(char)
            else:
                idx = ord(char) - ord('a') + 10
            label_tensor[i, idx] = 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_tensor


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
train_dataset = CaptchaDataset(image_folder=image_folder, image_files=train_files, transform=transform)
test_dataset = CaptchaDataset(image_folder=image_folder, image_files=test_files, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import torch.nn as nn

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 8, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 5 * 36)  
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x.view(-1, 5, 36)  

from torch.optim import Adam

model = CaptchaModel()
criterion = nn.BCEWithLogitsLoss()  
optimizer = Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from tqdm import tqdm

num_epochs = 50

def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()  
    total_loss = 0

    #print("\nPredicted vs True Labels:")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            for i in range(labels.size(0)): 
                predicted_label = decode_label(outputs[i])
                true_label = decode_label(labels[i])
                #print(f"Predicted: {predicted_label}, True: {true_label}")

                if predicted_label == true_label:
                    correct += 1
                total += 1

    accuracy = correct / total * 100
    print(f"\nTest Loss: {total_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

def decode_label(output):
    characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    decoded = ""
    for char_probs in output:
        index = char_probs.argmax().item()  
        decoded += characters[index]
    return decoded


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # forward
        outputs = model(images)
        
        # loss
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}")
    
    evaluate_model(model, test_loader, device)

model.eval()

torch.save(model.state_dict(), "captcha_model.pth")
with torch.no_grad():
    for images, labels in train_loader:  
        images = images.to(device)
        outputs = model(images)
        
        for i in range(5):  
            predicted_label = decode_label(outputs[i])
            #print(f"Predicted: {predicted_label}")


