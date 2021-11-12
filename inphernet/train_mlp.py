


import os, sys, json
import numpy as np
import pandas as pd
import joblib

import torch
from datetime import datetime
from data import GeneMeshData, GeneMeshMLPDataset
from models import MLP

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from utils import add_train_args


# argument parser
args = add_train_args()
torch.manual_seed(seed=123456)

os.makedirs(args.outdir, exist_ok=True)
tb = SummaryWriter(log_dir = args.outdir)


# Parameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_size = 1900 + 768
learning_rate = 0.001
num_epochs = 1000


# Load mesh embeddings
print("Load embeddings")
mesh_features = pd.read_csv(args.mesh_embed, index_col=0)
gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
gene_features.index = gene_features.index.astype(str)

train_data = joblib.load(os.path.join(args.outdir,"train.data.pkl"))
valid_data = joblib.load(os.path.join(args.outdir,"val.data.pkl"))


train_edge = torch.vstack([train_data['gene','genemesh','mesh'].edge_label_index, 
                          train_data['gene','genemesh','mesh'].edge_label]).T
valid_edge = torch.vstack([valid_data['gene','genemesh','mesh'].edge_label_index, 
                           valid_data['gene','genemesh','mesh'].edge_label]).T


with open(args.node2index, 'r') as j:
    node2index = json.load(j)
idx2gene = {str(i): v for v, i in node2index['gene2index'].items() }
idx2mesh = {str(i): v for v, i in node2index['mesh2index'].items() }


train_data = GeneMeshMLPDataset(train_edge, gene_features, mesh_features, idx2gene, idx2mesh)
valid_data = GeneMeshMLPDataset(valid_edge, gene_features, mesh_features, idx2gene, idx2mesh)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000, num_workers=1)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10000, num_workers=1)



print("Build Model", file=sys.stderr)
model = MLP(input_size)
model.to(device)
print(model)
# weight = [class_sample_count[0] / class_sample_count[1]]
# weight = [430489864/10800307]
criterion = torch.nn.BCEWithLogitsLoss() #
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate, 
                            momentum=0.9, weight_decay=0.0005)  
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)

ckpts = os.path.join(args.outdir, "mlp_best_model.pt")
if os.path.exists(ckpts):
    print("Loading checkpoints...", datetime.now(), file=sys.stderr)
    checkpoint = torch.load(ckpts)
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...', file=sys.stderr)
    # load trained optimizer state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Previously trained optimizer state_dict loaded...', file=sys.stderr)
    epoch_start = checkpoint['best_epoch']+1
    # load the criterion
    # criterion = checkpoint['loss']
    print('Trained model loss function loaded...', file=sys.stderr)
else:
    epoch_start = 1
    print("Train Model from very begining")


print("Start training: ",  datetime.now(), file=sys.stderr)
num_epochs += epoch_start
last_valid_loss = np.Inf

# Training the Model
for epoch in range(epoch_start, num_epochs):
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    for i, embeds  in enumerate(train_loader):
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
        #if (i) % 500 == 499:    # print every 500 mini-batches
        print('Current Time = %s, [%d, %5d] loss: %.7f' %
                  (datetime.now(), epoch, i + 1, loss.item()), file=sys.stderr)

    train_loss /= len(train_loader)               
    # # Test the Model
    model.eval()
    valid_loss = 0
    acc = 0
    auc = 0
    apr = 0
    correct = 0
    with torch.no_grad():
        for embeds in valid_loader:
            inputs, targets = embeds['embed'], embeds['target']
            inputs = inputs.to(device)
            targets = targets.view(-1).to(device)
            #targets = targets.view(-1).type(torch.int)
            outputs = model(inputs)
            valid_loss += criterion(outputs, targets).item() # 
            y_pred = torch.sigmoid(outputs.data).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            #fpr, tpr, thresholds = roc_curve(targets, y_pred, pos_label=1)
            #precision, recall, pr_threshold = precision_recall_curve(targets, y_pred, pos_label=1)
            auc += roc_auc_score(targets, y_pred)
            acc += accuracy_score(targets > 0, y_pred > 0.5)
            apr += average_precision_score(targets, y_pred)
    valid_loss /= len(valid_loader)
    auc /= len(valid_loader)
    acc /= len(valid_loader)
    apr /= len(valid_loader)
    print('Accuracy of the model on the test set: accuracy %d, AUC: %d %%' % (acc, auc))
    tb.add_scalar('Train/loss', train_loss, epoch) 
    tb.add_scalar('Valid/loss', valid_loss, epoch) 
    tb.add_scalar('Valid/pr', apr, epoch) 
    tb.add_scalar('Valid/roc', auc, epoch) 

    if valid_loss < last_valid_loss:
        last_test_loss = min(valid_loss, last_test_loss)
        # Save checkpoint
        PATH = f'./checkpoints/mlp_best_model.pt'
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, PATH)
        print(f"Save checkpoint to {PATH}", file=sys.stderr)
