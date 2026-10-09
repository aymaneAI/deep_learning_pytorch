import torch
from torch import nn
import requests
from pathlib import Path
from tqdm.auto import tqdm
from timeit import default_timer as timer
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Download single image
def download_single_image(image_path:str,
                          raw_url):
  """
  image_path must ends with .jpg
  """
  data_path = Path('data/')
  image_path = data_path/image_path
  if not image_path.is_file():
    with open(image_path, 'wb') as f:
      print(f'Download {image_path}...')
      request = requests.get(raw_url)
      f.write(request.content)
      print(f'{image_path} downloaded.')
  else:
    print(f'{image_path} already exist, skip the download.')
  return image_path


# Create the model for predict and plot single image
def predict_and_plot(image_path,
                     model,
                     transform,
                     class_names,
                     device = device):
  img = Image.open(image_path)
  transformed_image = transform(img)
  model.eval()
  with torch.inference_mode():
    x_logit = model(transformed_image.unsqueeze(0).to(device))
    pred_prob = torch.softmax(x_logit ,dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    class_name = class_names[pred_label.item()]
  print(f'Pred:{class_name}|Pred_prob:{pred_prob.max().item():.4f}')
  plt.figure(figsize=(10,6))
  plt.imshow(img)
  plt.title(f'Pred:{class_name}|Pred_prob:{pred_prob.max().item():.4f}')
  plt.axis(False);

# Predict all the samples
def predict_on_image(data_dir,
                     model,
                     class_names,
                     transform
                     device = device):
  pred_list = []
  paths = list(Path(data_dir).glob('*/*.jpg'))
  for path in paths:
    pred_dict = {}
    pred_dict['image_path'] = path
    class_label = path.parent.stem
    pred_dict['class_label'] = class_label
    img = Image.open(path)
    transformed_img = transform(img).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
      x_logits = model(transformed_img)
      pred_prob = torch.softmax(x_logits, dim = 1)
      pred_label = torch.argmax(pred_prob, dim = 1)

      pred_dict['pred_prob'] = pred_prob.max().cpu().item()
      pred_dict['pred_class'] = class_names[pred_label]
      pred_dict['correct'] = class_label == class_names[pred_label]
    pred_list.append(pred_dict)
  return pred_list


# Save model
def save_model(target_dir,
               model_name,
               model):
  TARGET_DIR = Path(f'{target_dir}/')
  TARGET_DIR.mkdir(parents = True, exist_ok = True)

  assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model_name shoud end with .pth or .pt'
  MODEL_SAVE_PATH = TARGET_DIR / model_name

  print(f'Saving model to: {MODEL_SAVE_PATH}')
  torch.save(obj = model.state_dict(), f=MODEL_SAVE_PATH)
  return MODEL_SAVE_PATH


# Download model
def download_model(loaded_model,
                   model_save_path,
                   device = device):
  loaded_model.load_state_dict(torch.load(f=model_save_path))
  loaded_model = loaded_model.to(device)
  return loaded_model

#Plot loss and accuracy curves
def plot_loss_acc_curves(model_results):
  train_loss = model_results['train_loss']
  train_acc = model_results['train_acc']
  test_loss = model_results['test_loss']
  test_acc = model_results['test_acc']
  n_epochs = range(len(train_loss))

  plt.figure(figsize=(12,5))
  # Curve 1
  plt.subplot(1,2,1)
  plt.plot(n_epochs, train_loss, label = 'train_loss')
  plt.plot(n_epochs, test_loss, label = 'test_loss')
  plt.legend()
  # Curve 2
  plt.subplot(1,2,2)
  plt.plot(n_epochs, train_acc, label = 'train_acc')
  plt.plot(n_epochs, test_acc, label = 'test_acc')
  plt.legend();
