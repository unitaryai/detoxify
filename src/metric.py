
import torch.nn.functional as F
import torch


def binary_cross_entropy(input, meta):

    if 'weight' in meta:
        target = meta['target'].to(input.device).reshape(input.shape)
        weight = meta['weight'].to(input.device).reshape(input.shape)
        return F.binary_cross_entropy_with_logits(input, target, weight=weight)
    elif 'multi_target' in meta:
        target = meta['multi_target'].to(input.device)
        loss_fn = F.binary_cross_entropy_with_logits
        loss = 0
        if 'class_weights' in meta:
            class_weights = meta['class_weights'][0]
        else:
            class_weights = torch.ones(target.shape[-1])/target.shape[-1]
        for i in range(target.shape[-1]):
            mask = target[:, i] != -1
            if torch.sum(mask) > 0:
                loss = loss + loss_fn(input[mask, i], target[:,i][mask].float()) * class_weights[i]
        return loss 
    else:
        target = meta['target'].to(input.device).reshape(input.shape)
        return F.binary_cross_entropy_with_logits(input, target)


def jigsaw_bias_binary_cross_entropy(input, meta):
    
    assert 'multi_target' in meta
    
    target = meta['multi_target'].to(input.device)
    meta['weights'] = meta['weights'].to(input.device)
    loss_fn = F.binary_cross_entropy_with_logits
    toxicity_loss = 0
    identity_loss = 0
    cnt_tox = 0
    cnt_idn = 0
    for i in range(target.shape[-1]):
        mask = target[:, i] != -1
        if torch.sum(mask) > 0:
            if i < 7:
                toxicity_loss = toxicity_loss + (loss_fn(input[mask, i], target[:,i][mask].float(), reduction='none') * meta['weights']).mean()
                cnt_tox += 1
            else:
                identity_loss = identity_loss + loss_fn(input[mask, i], target[:,i][mask].float()) 
                cnt_idn += 1

    toxicity_loss = toxicity_loss/cnt_tox
    identity_loss = identity_loss/cnt_idn if cnt_idn != 0 else identity_loss
    loss = toxicity_loss * 0.75 + identity_loss * 0.25
    return loss 


def binary_accuracy(output, meta, *args, **kwargs):
    if 'multi_target' in meta:
        correct = 0 
        target = meta['multi_target'].to(output.device)
        cnt = 0
        with torch.no_grad():
            for i in range(target.shape[-1]):
                mask = target[:, i] != -1
                pred = torch.sigmoid(output[mask,i]) > 0.5
                correct = correct + torch.sum(pred.to(output.device) == target[mask,i]).item()
                cnt = cnt + torch.sum(mask)
        return torch.tensor(correct / cnt.item())
    else:
        target = meta['target'].to(output.device).reshape(output.shape) > 0.5
        with torch.no_grad():
            pred = torch.sigmoid(output) > 0.5
            correct = torch.sum(pred == target).item()
        return torch.tensor(correct / len(target))