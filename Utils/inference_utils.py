import torch

def model_feedforward(model, data_loader, device, is_eval=True):
    if is_eval:
        model.eval()
    else:
        model.train()
    logits = []
    y_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            logits.extend(outputs)
            y_labels.extend(labels.to(device))
    return torch.stack(logits), torch.stack(y_labels)