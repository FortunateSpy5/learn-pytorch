from tqdm.auto import tqdm
import torch

def train_model(epochs, model, criterion, optimizer, train_loader, test_loader, device, scheduler=None):
    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    
    training_pbar = tqdm(range(epochs), unit="epochs")
    
    for epoch in training_pbar:
        training_pbar.set_description(f"Epoch {str(epoch+1).zfill(len(str(epochs)))}/{epochs}")
        iter_pbar = tqdm(total=len(train_loader)+len(test_loader), desc=f"Epoch {str(epoch+1).zfill(len(str(epochs)))}/{epochs}", unit="batches")

        train_correct = 0
        train_iter_loss = 0.0
        test_correct = 0
        test_iter_loss = 0.0
        
        model.train() # Set model to training mode (for dropout and batchnorm)
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            train_iter_loss += loss.item()
            # Get predictions from the maximum value
            predicted = torch.argmax(outputs, dim=1)
            train_correct += (predicted == labels).sum().item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_pbar.update(1)
        
        if scheduler:
            scheduler.step()
        
        model.eval() # Set model to evaluation mode (for dropout and batchnorm)
        for i, (images, labels) in enumerate(test_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            test_iter_loss += loss.item()
            # Get predictions from the maximum value
            predicted = torch.argmax(outputs, dim=1)
            test_correct += (predicted == labels).sum().item()
            iter_pbar.update(1)
        
        metrics["train_loss"].append(train_iter_loss/len(train_loader))
        metrics["train_accuracy"].append(train_correct/len(train_loader.dataset))
        metrics["test_loss"].append(test_iter_loss/len(test_loader))
        metrics["test_accuracy"].append(test_correct/len(test_loader.dataset))
        
        training_pbar.set_postfix(dict(zip(metrics.keys(), [i[-1] for i in metrics.values()])))
        iter_pbar.set_postfix(dict(zip(metrics.keys(), [i[-1] for i in metrics.values()])))
        iter_pbar.close()
    training_pbar.close()
    return metrics