from torch import nn, ones_like, zeros_like

def real_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = ones_like(disc_pred)
    loss = criterion(disc_pred, ground_truth)

    return loss

def fake_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = zeros_like(disc_pred)
    loss = criterion(disc_pred, ground_truth)

    return loss