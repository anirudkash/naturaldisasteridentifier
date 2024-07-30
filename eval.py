# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def preprocess_image(image_path):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = preprocess(image).unsqueeze(0)
#     return image

# class FeatureExtractor:
#     def __init__(self, model):
#         self.model = model
#         self.feature_maps = []
#         self.gradients = []
#         self.hook = model.layer4[1].register_forward_hook(self.hook_fn)
#         self.hook_backward = model.layer4[1].register_full_backward_hook(self.hook_backward_fn)

#     def hook_fn(self, module, input, output):
#         self.feature_maps = output

#     def hook_backward_fn(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def get_features(self):
#         return self.feature_maps

#     def get_gradients(self):
#         return self.gradients

# def generate_cam(model, image_path, device):
#     extractor = FeatureExtractor(model)
#     image = preprocess_image(image_path)
#     image = image.to(device)
#     output = model(image)

#     # Get the index of the class with the highest score
#     class_idx = output.argmax().item()
#     class_scores = nn.functional.softmax(output,dim=1)
#     class_confidence = class_scores[0, class_idx].item()

#     # Backward pass to get gradients
#     model.zero_grad()
#     output[0, class_idx].backward()

#     # Get the gradients and feature maps from the hooks
#     gradients = extractor.get_gradients()
#     activations = extractor.get_features()

#     # Ensure tensors are on the same device
#     gradients = gradients.to(device)
#     activations = activations.to(device)

#     # Compute the weights
#     weights = torch.mean(gradients, dim=[0, 2, 3])
#     cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=device)

#     for i, w in enumerate(weights):
#         cam += w * activations[0, i, :, :]

#     cam = F.relu(cam)
#     cam -= cam.min()
#     cam /= cam.max()
#     cam = cam.cpu().detach().numpy()

#     # Load and preprocess the original image
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))

#     # Resize CAM to match original image size
#     cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
#     cam = np.uint8(255 * cam)
#     cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#     print('CAM.shape: ', cam.shape)
#     # Overlay CAM on the original image
#     cam = np.float32(cam) + np.float32(image)
#     cam = 255 * cam / np.max(cam)
#     cam = np.uint8(cam)

#     return cam, class_idx, class_confidence

# def main():

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     print('Defined the transforms adn tensors')

#     # Load a pre-trained ResNet model and modify it for the number of classes in your dataset
#     num_classes = 6  # Set the number of classes
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated for deprecation warning
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     print('Loadedd in model')

#     global device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('Device being used: ', device)
#     model.to(device)
#     print('setting the device')

#     # Load the trained weights
#     model.load_state_dict(torch.load('resnet18_weights.pth', map_location=device))
#     model.eval()  # Set the model to evaluation mode
#     print('loaded trained weights')

#     cam_image, predicted_class, confidence = generate_cam(model, r'C:\Users\aniru\OneDrive\Desktop\NaturalDisaster\Comprehensive Disaster Dataset (CDD)\Comprehensive Disaster Dataset(CDD)\Fire_Disaster\Wild_Fire\01_02_0004.png', device)

#     plt.figure(figsize=(12,8))
#     plt.subplot(231)
#     plt.imshow(cv2.imread(r'C:\Users\aniru\OneDrive\Desktop\NaturalDisaster\Comprehensive Disaster Dataset (CDD)\Comprehensive Disaster Dataset(CDD)\Fire_Disaster\Wild_Fire\01_02_0004.png', cv2.COLOR_BGR2RGB))
#     plt.subplot(232)
#     plt.imshow(cam_image, cmap='gray')
#     plt.show()

#     dataset = datasets.ImageFolder(root=r'C:\Users\aniru\OneDrive\Desktop\NaturalDisaster\Comprehensive Disaster Dataset (CDD)\Comprehensive Disaster Dataset(CDD)', transform=transform)
#     class_name = dataset.classes
#     print(class_name)
#     print('predicted_class: ', predicted_class)
#     print(f'Image classified as: {class_name[predicted_class]}; confidence: {confidence:.2f}')
    


# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []
        self.hook = model.layer4[1].register_forward_hook(self.hook_fn)
        self.hook_backward = model.layer4[1].register_full_backward_hook(self.hook_backward_fn)

    def hook_fn(self, module, input, output):
        self.feature_maps = output

    def hook_backward_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_features(self):
        return self.feature_maps

    def get_gradients(self):
        return self.gradients

def generate_cam(model, image_path, device):
    extractor = FeatureExtractor(model)
    image = preprocess_image(image_path)
    image = image.to(device)
    output = model(image)

    # Get the index of the class with the highest score
    class_idx = output.argmax().item()
    class_scores = nn.functional.softmax(output, dim=1)
    class_confidence = class_scores[0, class_idx].item()

    # Backward pass to get gradients
    model.zero_grad()
    output[0, class_idx].backward()

    # Get the gradients and feature maps from the hooks
    gradients = extractor.get_gradients()
    activations = extractor.get_features()

    # Ensure tensors are on the same device
    gradients = gradients.to(device)
    activations = activations.to(device)

    # Compute the weights
    weights = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=device)

    for i, w in enumerate(weights):
        cam += w * activations[0, i, :, :]

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()

    # Load and preprocess the original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Resize CAM to match original image size
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)

    # **Threshold the CAM to find regions of interest**
    threshold = 0.4 * cam.max()
    cam_bin = np.where(cam > threshold, 255, 0).astype(np.uint8)

    # **Find contours in the thresholded CAM**
    contours, _ = cv2.findContours(cam_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **Draw bounding boxes around the contours**
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Overlay CAM on the original image
    cam_colormap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam_overlay = np.float32(cam_colormap) + np.float32(image)
    cam_overlay = 255 * cam_overlay / np.max(cam_overlay)
    cam_overlay = np.uint8(cam_overlay)

    return cam_overlay, class_idx, class_confidence

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print('Defined the transforms')

    # Load a pre-trained ResNet model and modify it for the number of classes in your dataset
    num_classes = 6  # Set the number of classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated for deprecation warning
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print('Loaded pre-trained ResNet model')

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device being used: ', device)
    model.to(device)
    print('Setting the Device')

    # Load the trained weights
    model.load_state_dict(torch.load('resnet18_weights.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode
    print('Loaded trained weights')

    # Load the dataset and split it into training and validation sets
    # dataset = datasets.ImageFolder(root=r'C:\Users\aniru\OneDrive\Desktop\NaturalDisaster\Comprehensive Disaster Dataset (CDD)\Comprehensive Disaster Dataset(CDD)\Fire_Disaster', transform=transform)
    dataset = datasets.ImageFolder(root=r'C:\Users\aniru\OneDrive\Desktop\NaturalDisaster\GQ Damage photos', transform=transform)
    class_names = dataset.classes
    print('Class names:', class_names)

    # Split the dataset into training and validation sets
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Test some images from the validation set
    for i, (inputs, labels) in enumerate(val_loader):
        # if i == 5:  # Test on 5 images
        #     break

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Save the input image for CAM generation
        img_path = val_dataset.dataset.imgs[val_dataset.indices[i]][0]
        cam_image, predicted_class, confidence = generate_cam(model, img_path, device)

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.subplot(232)
        plt.imshow(cam_image, cmap='gray')
        plt.show()

        print(f'Image classified as: {class_names[predicted_class]}. Confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()

