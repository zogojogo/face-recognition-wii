import os
import torch
import torchvision as vision
from PIL import Image
import io
import numpy as np
    
def extract_feature(path, model, mode):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  transform = vision.transforms.Compose([
                vision.transforms.Resize((160, 160)), 
                vision.transforms.ToTensor(),
                vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
  model.eval().to(device)
  if mode == 'api':
    img = Image.open(io.BytesIO(path))
  else:
    img = Image.open(path)
  img_transformed = transform(img).unsqueeze(0)
  img_embedding = model(img_transformed.to(device)).cpu().detach().numpy()
  return img_embedding

def Var(x:np.ndarray):
    n = len(x)
    x_hat = x.mean()
    return n * np.sum((x-x_hat)**2)

def NED(u,v):
    return 0.5 * Var(u-v) / (Var(u) + Var(v))

def embed2dict(data_path, model, mode='normal'):
    embeddings = {}
    for folder in os.listdir(data_path):
        for file in os.listdir(data_path + folder):
            img_path = data_path + folder + "/" + file
            img_embedding = extract_feature(img_path, model, mode)
            embeddings[img_path] = img_embedding
    return embeddings