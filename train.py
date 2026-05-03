import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, train_loader, test_loader, device, args):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    writer = SummaryWriter(log_dir=args.log_dir)

    model.train()
    is_pointnet2 = args.model == 'pointnet2'

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

            if is_pointnet2:
                points = batch['points'].permute(0, 2, 1)  # [B, 3, N]
                logits, _ = model(points)
                loss = criterion(logits, batch['label'].long())
            else:
                # OCNN: forward takes data, octree, depth
                logits, _ = model(batch)
                loss = criterion(logits, batch['label'].long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_samples += batch['label'].size(0)
            total_correct += (predicted == batch['label']).sum().item()

            # Update pbar with loss and acc
            avg_loss = total_loss / (pbar.n + 1)
            acc = 100 * total_correct / total_samples
            pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{acc:.2f}%')

        # Validate with test set
        acc = evaluate(model, test_loader, device, args)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{args.model}_best_model.pth')
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {acc}')

        # tensorboard log
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)

    writer.close()

def evaluate(model, dataloader, device, args):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    is_pointnet2 = args.model == 'pointnet2'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

            if is_pointnet2:
                points = batch['points'].permute(0, 2, 1)  # [B, 3, N]
                logits, _ = model(points)
            else:
                data = model.get_input_feature(batch['octree'])
                logits = model.model(data, batch['octree'], batch['octree'].depth)

            loss = criterion(logits, batch['label'].long())
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}')
    return accuracy