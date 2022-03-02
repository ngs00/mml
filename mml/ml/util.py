import numpy
import torch
import xgboost as xgb
from sklearn.metrics import r2_score


def normalize(data, f_min, f_max):
    return (data - f_min) / (f_max - f_min)


def split_dataset(dataset, ratio, random_seed):
    n_data = len(dataset)
    n_dataset1 = int(ratio * n_data)

    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(n_data)

    dataset1 = [dataset[idx] for idx in idx_rand[:n_dataset1]]
    dataset2 = [dataset[idx] for idx in idx_rand[n_dataset1:]]

    return dataset1, dataset2


def even_samples(min, max, n_samples):
    samples = numpy.empty(n_samples)
    len = (max - min) / n_samples

    for i in range(0, n_samples):
        samples[i] = min + len * (i + 1)

    return samples


def rbf(x, mu, beta):
    return numpy.exp(-(x - mu)**2 / beta**2)


def train(model, data_loader, optimizer, criterion):
    model.train()
    train_loss = 0

    for batch in data_loader:
        batch.cuda()

        preds = model(batch)
        loss = criterion(batch.y, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()

            preds = model(batch)
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0)


def train_xgb(train_x, train_y, test_x, test_y):
    min_val_error = 1e+8
    opt_d = -1
    opt_n = -1
    n_train = int(0.8 * train_x.shape[0])
    val_x = train_x[n_train:, :]
    val_y = train_y[n_train:, :]
    train_x = train_x[:n_train, :]
    train_y = train_y[:n_train, :]
    train_y = train_y.reshape(train_y.shape[0], -1)
    test_y = test_y.reshape(test_y.shape[0], -1)

    for d in range(3, 10):
        for n in [100, 150, 200, 300, 400]:
            model_xgb = xgb.XGBRegressor(max_depth=d, n_estimators=n)
            model_xgb.fit(train_x, train_y, eval_metric='mae')
            pred_test = model_xgb.predict(val_x).reshape(-1, 1)
            val_error = numpy.mean(numpy.abs(val_y - pred_test))
            print('d={}\tn={}\tMAE: {:.4f}'.format(d, n, val_error))

            if val_error < min_val_error:
                min_val_error = val_error
                opt_d = d
                opt_n = n

    model_xgb = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n)
    model_xgb.fit(train_x, train_y, eval_metric='mae')
    pred_test = model_xgb.predict(test_x).reshape(-1, 1)
    test_mae = numpy.mean(numpy.abs(test_y - pred_test))
    print('opt d={}\topt n={}\tmin MAE: {:.4f}'.format(opt_d, opt_n, test_mae))
    r2 = r2_score(pred_test, test_y)
    print('R2: ' + str(r2))
