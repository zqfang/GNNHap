'''
The LinkPredictor (MLP) for GeneMesh HeteroGraph training
'''

import os, sys, json
import numpy as np
import pandas as pd
import joblib

import torch
from datetime import datetime
from data import GeneMeshData, GeneMeshMLPDataset
from models import MLP

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, recall_score
from utils import add_train_args
from tqdm.auto import tqdm

# argument parser
args = add_train_args()
torch.manual_seed(seed=123456)
os.makedirs(args.outdir, exist_ok=True)
# Parameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_size = 1900 + 768
#input_size = args.input_size 
hidden_size = args.hidden_size # 64
num_epochs = args.num_epochs # 100
batch_size = args.batch_size # 50000
learning_rate = args.lr # 0.01
num_workers = args.num_cpus # 6
one_hot = True
# TB
tb = SummaryWriter(log_dir = os.path.join(args.outdir, "log/mlp"), filename_suffix=f".MLP_{hidden_size}_onehot")
# Load mesh embeddings
print("Load embeddings")
if one_hot:
    mesh_features = None
    gene_features = None
else:
    mesh_features = None #pd.read_csv(args.mesh_embed, index_col=0, header=None)
    gene_features = None #pd.read_csv(args.gene_embed, index_col=0, header=None)
    gene_features.index = gene_features.index.astype(str)

train_data = joblib.load(os.path.join(args.outdir,"train.data.pkl"))
valid_data = joblib.load(os.path.join(args.outdir,"val.data.pkl"))


train_edge = torch.vstack([train_data['gene','genemesh','mesh'].edge_label_index, 
                          train_data['gene','genemesh','mesh'].edge_label]).T
valid_edge = torch.vstack([valid_data['gene','genemesh','mesh'].edge_label_index, 
                           valid_data['gene','genemesh','mesh'].edge_label]).T

perm = torch.randperm(train_edge.shape[0])
train_edge = train_edge[perm,:].type(torch.LongTensor)
## No need to permute validation data, but it's fine
perm = torch.randperm(valid_edge.shape[0])
valid_edge = valid_edge[perm,:].type(torch.LongTensor)

## if use onehot
if one_hot:
    input_size = len(train_data['nid2gene']) + len(train_data['nid2mesh'])

print("DataLoader")
train_data = GeneMeshMLPDataset(train_edge, train_data['nid2gene'], train_data['nid2mesh'], gene_features, mesh_features)
valid_data = GeneMeshMLPDataset(valid_edge, valid_data['nid2gene'], valid_data['nid2mesh'], gene_features, mesh_features)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)

print("Build Model")
model = MLP(input_size, hidden_size)
model.to(device)
print(model)
# weight = [class_sample_count[0] / class_sample_count[1]]
# weight = [430489864/10800307]
criterion = torch.nn.BCEWithLogitsLoss() #
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate, 
                            momentum=0.9, weight_decay=0.0005)  
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=2, eta_min=1e-4)
'''
以T_0=5, T_mult=1为例:
T_0:学习率第一次回到初始值的epoch位置.
T_mult:这个控制了学习率回升的速度
    - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
        - 5,10,15,20,25,.......处回到最大值
    - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
        - 5,15,35,75,155,.......处回到最大值
'''

epoch_start = 0
weight = "mlp_{hidden_size}_best_model.pt"
if one_hot: weight = "mlp_{hidden_size}_onehot_best_model.pt"
ckpts = os.path.join(args.outdir, weight)
if os.path.exists(ckpts):
    tqdm.write(f"Loading checkpoints .... {datetime.now():%Y-%m-%d %H:%M:%S}")
    checkpoint = torch.load(ckpts)
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    tqdm.write('Previously trained model weights state_dict loaded...')
    # load trained optimizer state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    tqdm.write('Previously trained optimizer state_dict loaded...')
    epoch_start = checkpoint['epoch']+1
    # load the criterion
    # criterion = checkpoint['loss']
    tqdm.write('Trained model loss function loaded...')

tqdm.write(f"Loading checkpoints .... {datetime.now():%Y-%m-%d %H:%M:%S}")
 
epoch_start = min(epoch_start, num_epochs)
last_valid_loss = np.Inf

# Training the Model
for epoch in tqdm(range(epoch_start, num_epochs), total=num_epochs, desc='Epoch', position=0,):
    model.train()
    tqdm.write(f"Training epoch: {epoch}, time: {datetime.now()}")
    train_loss = 0.0
    for batch, embeds  in enumerate(tqdm(train_loader, total=len(train_loader), desc='Train', position=1, leave=True)):
        inputs, targets = embeds['embed'], embeds['target']
        inputs = inputs.to(device)
        targets = targets.view(-1).to(device)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        # target size must be the same as ouput size
        loss = criterion(outputs, targets) # 
        loss.backward()
        optimizer.step()
        # print statistics
        train_loss += loss.item()
        '''
        scheduler.step(epoch + batch / len(train_loader))的理解如下,
        如果是一个epoch结束后再.step 那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
        则可以在这里进行.step
        '''
        lr = scheduler.get_last_lr()[0]
        scheduler.step(epoch + batch / len(train_loader))
      
        tqdm.write('%s, epoch %d, step %5d, loss %.7f, lr %.7f' %
                  (datetime.now(), epoch, batch + 1, loss.item(), lr))

    train_loss /= len(train_loader)       
    # # Test the Model
    model.eval()
    valid_loss = 0
    acc = 0
    auc = 0
    apr = 0
    correct = 0
    y = []
    y_preds = []

    PATH = os.path.join(args.outdir, f'mlp_{hidden_size}_epoch{epoch:03d}.pt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, PATH)

    tqdm.write(f"Validation epoch: {epoch}, time: {datetime.now()}")
    with torch.no_grad():
        for embeds in tqdm(valid_loader, total=len(valid_loader), desc='Train', position=1, leave=True):
            inputs, targets = embeds['embed'], embeds['target']
            inputs = inputs.to(device)
            targets = targets.view(-1).to(device)
            #targets = targets.view(-1).type(torch.int)
            outputs = model(inputs)
            valid_loss += criterion(outputs, targets).item() # 
            y_pred = torch.sigmoid(outputs.data).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            y_preds.append(y_pred)
            y.append(targets)
    # save model
    valid_loss /= len(valid_loader)
    if valid_loss < last_valid_loss:
        last_valid_loss = min(valid_loss, last_valid_loss)
        # Save checkpoint
        PATH = os.path.join(args.outdir, weight)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, PATH)
        tqdm.write(f"Save checkpoint to {PATH}")

    # other metric
    y_preds = np.concatenate(y_preds)
    y = np.concatenate(y)       
    auc = roc_auc_score(y, y_preds)
    acc = accuracy_score(y > 0, y_preds > 0.5)
    rs = recall_score(y > 0, y_preds > 0.5)
    apr = average_precision_score(y, y_preds)
    tqdm.write(f'Validation: {datetime.now():%Y-%m-%d %H:%M:%S}, epoch {epoch:4d}, accuracy {acc:.2f}, pr {apr:.2f}, auc {auc:.2f}, recall {rs:.2f} ')
    tb.add_scalars('Loss', {'train':train_loss,
                             'valid':valid_loss}, epoch) 
    tb.add_scalars('Valid',{'ap':apr, 'auc': auc,'recall': rs}, epoch) 
    tb.add_pr_curve("Precision-Recall", y, y_preds, epoch)


tb.add_graph(model, inputs)
tb.close()
