import os
import torch
import clip
from PIL import Image
from torchvision.datasets import CIFAR100
import numpy as np
from tqdm import tqdm
# from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) #返回的是clip模型 图像预处理操作


""" demo1 """
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device) #-》torch.Size([1, 3, 224, 224])
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) #-》torch.Size([3, 77])

with torch.no_grad():
    image_features = model.encode_image(image) #torch.Size([1, 512])
    text_features = model.encode_text(text) #torch.Size([3, 512])
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


""" demo2 """
# # Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# # Prepare the inputs
# image, class_id = cifar100[3637]
# image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)

# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# values, indices = similarity[0].topk(5)

# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
    
    
    
# """ demo3 """
# # Load the dataset
# root = os.path.expanduser("~/.cache")
# train = CIFAR100(root, download=True, train=True, transform=preprocess)
# test = CIFAR100(root, download=True, train=False, transform=preprocess)


# def get_features(dataset):
#     all_features = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
#             features = model.encode_image(images.to(device))

#             all_features.append(features)
#             all_labels.append(labels)

#     return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# # Calculate the image features
# train_features, train_labels = get_features(train)
# test_features, test_labels = get_features(test)

# # Perform logistic regression
# classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
# classifier.fit(train_features, train_labels)

# # Evaluate using the logistic regression classifier
# predictions = classifier.predict(test_features)
# accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
# print(f"Accuracy = {accuracy:.3f}")