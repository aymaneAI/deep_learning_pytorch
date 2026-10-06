import torch
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm

def train_step(model,
               train_dataloader,
               loss_fn,
               optimizer,
               device):
  train_loss, train_acc = 0,0
  model = model.to(device)
  model.train()
  for batch,(X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    x_logits = model(X)
    x_preds = torch.argmax(x_logits, dim=1)
    loss = loss_fn(x_logits, y)
    train_loss += loss
    train_acc += (((y == x_preds).sum().item())/len(x_preds)) * 100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)
  return train_loss, train_acc

def test_step(model,
              test_dataloader,
              loss_fn,
              device):
  test_loss, test_acc = 0,0
  model = model.to(device)
  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)
      test_logits = model(X)
      test_preds = torch.argmax(test_logits, dim=1)
      test_loss += loss_fn(test_logits, y).item()
      test_acc += (((y == test_preds).sum().item()) / len(test_preds)) * 100
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    return test_loss, test_acc

def train(model,
                train_dataloader,
                test_dataloader,
                optimizer,
                loss_fn,
                epochs,
                device):

  results = {'train_loss':[],
               'train_acc':[],
               'test_loss':[],
               'test_acc':[]}
  start_train = timer()
  for epoch in tqdm(range(epochs)):
    print()
    print(f'Epoch {epoch+1} -------------')
    train_loss, train_acc = train_step(model = model,
                                       train_dataloader = train_dataloader,
                                       loss_fn = loss_fn,
                                       optimizer = optimizer,
                                       device = device)
    test_loss, test_acc = test_step(model = model,
                                    test_dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    device = device)
    print(f'train_loss : {train_loss:.4f} | train_acc : {train_acc:.2f}% | test_loss : {test_loss:.4f} | test_acc : {test_acc:.2f}%')
    
    results['train_loss'].append(train_loss.cpu())
    results['test_loss'].append(test_loss.cpu())
    results['train_acc'].append(train_acc.cpu())
    results['test_acc'].append(test_acc.cpu())
  end_train = timer()
  train_time = end_train - start_train
  print(f'The model was training for {train_time // 60} min and {train_time % 60:.4f} sec.')
  return results
