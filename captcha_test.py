import os
import torch
import cv2
import torchvision.transforms as transforms
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

model = CaptchaModel()

def decode_label(output):
    characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    decoded = ""
    for char_probs in output:
        index = char_probs.argmax().item()  
        decoded += characters[index]
    return decoded

def load_model(model_path, device):
    model = CaptchaModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Görselleri işleme
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 64))  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    image = transform(image)
    return image.unsqueeze(0)  

def predict_captcha(model, image_path, device):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted_label = decode_label(outputs[0])  
    return predicted_label


import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    accuracy = 0
    
    start_time = time.time()

    model_path = "captcha_model_97.pth" 
    model = load_model(model_path, device)
    
    test_images_folder = "captcha_images"  
    image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Tahmin yap
    for image_file in image_files:
        image_path = os.path.join(test_images_folder, image_file)
        predicted_text = predict_captcha(model, image_path, device)

        x = "False"

        if image_file.split(".")[0] == predicted_text:
            x = "True"
            accuracy += 1
        
        print(f"Image: {image_file}, Predicted Text: {predicted_text} -- {x}")


    print("Accuracy --> ", accuracy / len(image_files))

    end_time = time.time()

    print("total time:", end_time - start_time)

if __name__ == "__main__":
    main()
