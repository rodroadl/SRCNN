import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    num_val_batches = len(dataloader)
    net.eval()
    score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, label = batch['image'], batch['label']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device=device, dtype=torch.float32)

            # predict the mask
            pred = net(image)
            
    net.train()
    return score