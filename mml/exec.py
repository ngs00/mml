import joblib
import torch.utils.data as data
import torch_geometric.data as gdata
import ml.mml as mml
from xgboost import XGBRegressor
from ml.gnn import *
from ml.util import split_dataset


###########################################################################################
#                               Experiment settings                                       #
###########################################################################################
random_seed = 0
n_bond_feats = 128
batch_size = 16
init_lr = 5e-4
coeff_cos_sim = 1e-2
dim_emb = 32
mml.n_smp = batch_size


###########################################################################################
#               Model parameter optimization of the embedding network                     #
###########################################################################################
# Load dataset
dataset = load_dataset(path_structs='datasets/hoip',
                       metadata_file='datasets/hoip/metadata.xlsx',
                       idx_struct=0,
                       idx_target=1,
                       n_bond_feats=n_bond_feats)
dataset_train, dataset_test = split_dataset(dataset, ratio=0.8, random_seed=random_seed)
train_idx = numpy.array([x.idx for x in dataset_train]).reshape(-1, 1)
test_idx = numpy.array([x.idx for x in dataset_test]).reshape(-1, 1)
mml.set_y_vars(dataset)

# Create data loaders for graph-structured data.
loader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=mml.collate)
loader_emb = gdata.DataLoader(dataset_train, batch_size=batch_size)
loader_test = gdata.DataLoader(dataset_test, batch_size=batch_size)

# Define an embedding network of MML.
emb_net = TFNN(dataset[0].x.shape[1], n_bond_feats=n_bond_feats, dim_out=dim_emb).cuda()

# Define an optimizer to train the embedding network.
optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr)

# Train the embedding network.
for i in range(0, 300):
    train_loss = mml.train(emb_net, loader_train, optimizer, coeff_cos_sim=coeff_cos_sim)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, 300, train_loss))

# Save the trained embedding network.
torch.save(emb_net.state_dict(), 'results/hoip/emb_net.pt')

# Generate the embeddings of the crystal structures in the training dataset.
emb_train = mml.test(emb_net, loader_emb).cpu().detach().numpy()
train_targets = numpy.vstack([x.y.item() for x in dataset_train])

# Generate the embeddings of the crystal structures in the test dataset.
emb_test = mml.test(emb_net, loader_test).cpu().detach().numpy()
test_targets = numpy.vstack([x.y.item() for x in dataset_test])

# Save the embedding network and the embedding results.
numpy.savetxt('results/hoip/embs_train.csv', numpy.hstack([emb_train, train_targets, train_idx]), delimiter=',')
numpy.savetxt('results/hoip/embs_test.csv', numpy.hstack([emb_test, test_targets, test_idx]), delimiter=',')


###########################################################################################
#      Model parameter optimization of the prediction model based on neural network       #
###########################################################################################
# Load the generated embeddings of the training dataset.
dataset_train = numpy.array(pandas.read_csv('results/hoip/embs_train.csv', header=None))
data_train_x = torch.tensor(dataset_train[:, :dim_emb], dtype=torch.float)
data_train_y = torch.tensor(dataset_train[:, dim_emb], dtype=torch.float).view(-1, 1)
loader_train = data.DataLoader(data.TensorDataset(data_train_x, data_train_y), batch_size=64, shuffle=True)

# Load the generated embeddings of the test dataset.
dataset_test = numpy.array(pandas.read_csv('results/hoip/embs_test.csv', header=None))
data_test_x = torch.tensor(dataset_test[:, :dim_emb], dtype=torch.float)
data_test_y = torch.tensor(dataset_test[:, dim_emb], dtype=torch.float).view(-1, 1)
loader_test = data.DataLoader(data.TensorDataset(data_test_x, data_test_y), batch_size=128)

# Define a prediction model based on fully-connected neural network.
pred_model_fc = FNN(dim_in=dim_emb, dim_out=1).cuda()

# Define an optimizer to train the prediction network.
optimizer = torch.optim.Adam(pred_model_fc.parameters(), lr=5e-4, weight_decay=1e-7)

# Define a prediction loss.
criterion = torch.nn.L1Loss()

# Train MML-FC.
for epoch in range(0, 1000):
    train_loss = pred_model_fc.fit(loader_train, optimizer, criterion)
    print('Epoch [{}/{}]\tTrain Loss: {:.4f}'.format(epoch, 1000, train_loss))

# Save trained MML-FC.
torch.save(pred_model_fc.state_dict(), 'results/hoip/pred_model_fc.pt')

# Evaluate MML-FC on the test dataset.
targets_test = data_test_y.numpy()
preds_test = pred_model_fc.predict(loader_test)
test_mae_fc = mean_absolute_error(targets_test, preds_test)
test_r2_fc = r2_score(targets_test, preds_test)

# Save the prediction results of MML-FC.
numpy.savetxt('results/hoip/preds_fc.csv', numpy.hstack([test_idx, targets_test, preds_test]), delimiter=',')


###########################################################################################
#          Model parameter optimization of the prediction model based on XGBoost          #
###########################################################################################
# Load the generated embeddings of the training dataset.
dataset_train = numpy.array(pandas.read_csv('results/hoip/embs_train.csv', header=None))
data_train_x = dataset_train[:, :dim_emb]
data_train_y = dataset_train[:, dim_emb]

# Load the generated embeddings of the test dataset.
dataset_test = numpy.array(pandas.read_csv('results/hoip/embs_test.csv', header=None))
data_test_x = dataset_test[:, :dim_emb]
data_test_y = dataset_test[:, dim_emb]

# Define a prediction model based on gradient boosting tree regression.
pred_model_gb = XGBRegressor(max_depth=7, n_estimators=400)

# Train MML-GB.
pred_model_gb.fit(data_train_x, data_train_y)

# Save trained MML-GB.
joblib.dump(pred_model_gb, 'results/hoip/pred_model_gb.joblib')

# Evaluate MML-FC on the test dataset.
preds_test = pred_model_gb.predict(data_test_x)
test_mae_gb = mean_absolute_error(data_test_y, preds_test)
test_r2_gb = r2_score(data_test_y, preds_test)

# Save the prediction results of MML-FC.
pred_results = numpy.hstack([test_idx, targets_test.reshape(-1, 1), preds_test.reshape(-1, 1)])
numpy.savetxt('results/hoip/preds_fc.csv', pred_results, delimiter=',')


###########################################################################################
#            Print evaluation metrics of MML-FC and MML-GB on the test dataset            #
###########################################################################################

print('MAE of MML-FC: {:.4f}\tR2 of MML-FC: {:.4f}'.format(test_mae_fc, test_r2_fc))
print('MAE of MML-GB: {:.4f}\tR2 of MML-GB: {:.4f}'.format(test_mae_gb, test_r2_gb))
