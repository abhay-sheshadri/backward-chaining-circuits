import torch
from tqdm import tqdm


def get_loaders(dataset, batch_size, train_test_split=0.9):
    # Split the dataset into train and test sets
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                    [train_size, test_size])
    # Collate function
    def collate(data):
        tokens, masks = zip(*data)
        tokens = torch.stack(tokens, dim=0)
        masks = torch.stack(masks, dim=0)
        return tokens, masks
    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate)
    return train_loader, test_loader


def train(model, train_loader, test_loader, n_epochs, learning_rate=3e-4, betas=(0.9, 0.99)):
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, betas=betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 2e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Start training
    for epoch in range(n_epochs):

        pbar = tqdm(total=len(train_loader))
        model.train()
        losses = []
        accs = []

        for idx, (tokens, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            
            tokens = tokens.cuda().to(torch.long)
            inputs = tokens[:, :-1]
            output_mask = mask[:, 1:].cuda()
            targets = tokens[:, 1:][output_mask]
            
            outputs = model(inputs)[output_mask]
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            losses.append(loss.item())
            acc = torch.mean( (outputs.argmax(-1) == targets).to(torch.float)) * 100
            accs.append(acc)

            pbar.set_description(f"TRAIN - Epoch: {epoch+1}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
            pbar.update(1)

        scheduler.step()
        pbar.close()
    
        model.eval()
        with torch.no_grad():

            pbar = tqdm(total=len(test_loader))
            losses = []
            accs = []

            for idx, (tokens, mask) in enumerate(test_loader):

                tokens = tokens.cuda().to(torch.long)
                inputs = tokens[:, :-1]
                output_mask = mask[:, 1:].cuda()
                targets = tokens[:, 1:][output_mask]
                
                outputs = model(inputs)[output_mask]
                loss = loss_fn(outputs, targets)
                
                losses.append(loss.item())
                acc = torch.mean( (outputs.argmax(-1) == targets).to(torch.float)) * 100
                accs.append(acc)
                
                pbar.set_description(f"TEST  - Epoch: {epoch+1}, Loss: {sum(losses)/len(losses):.4f}, Acc: {sum(accs)/len(accs):.4f}%")
                pbar.update(1)
        
            pbar.close()