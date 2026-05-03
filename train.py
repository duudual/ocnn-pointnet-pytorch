import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, train_loader, test_loader, device, args):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_acc = 0.0

    writer = SummaryWriter(log_dir=args.log_dir)

    is_pointnet2 = args.model == 'pointnet2'

    for epoch in range(args.epochs):
        model.train()
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
                # PointNet2 forward already returns log_softmax, use nll_loss
                loss = F.nll_loss(logits, batch['label'].long())
                _, predicted = torch.max(logits, 1)
                correct = (predicted == batch['label']).sum().item()
            else:
                # OCNN: forward returns (loss, accu)
                loss, accu = model(batch)
                correct = int(accu.item() * batch['label'].size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += batch['label'].size(0)
            total_correct += correct

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
        scheduler.step()

    writer.close()

def evaluate(model, dataloader, device, args):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    is_pointnet2 = args.model == 'pointnet2'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

            if is_pointnet2:
                points = batch['points'].permute(0, 2, 1)  # [B, 3, N]
                logits, _ = model(points)
                loss = F.nll_loss(logits, batch['label'].long())
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch['label']).sum().item()
            else:
                # OCNN: forward returns (loss, accu)
                loss, accu = model(batch)
                correct += int(accu.item() * batch['label'].size(0))

            total_loss += loss.item()
            total += batch['label'].size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}')
    return accuracy