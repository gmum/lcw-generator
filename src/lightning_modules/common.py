from evaluators.fid_evaluator import FidEvaluator
import torch
from factories.dataset_factory import get_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def get_train_dataloader(args):
    dataset = get_dataset(args.dataset, args.dataroot, train=True)

    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=int(args.workers))


def get_val_dataloader(args, batch_size: int = None):
    if batch_size is None:
        batch_size = args.batch_size
    dataset = get_dataset(args.dataset, args.dataroot, train=False)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)


def convert_to_latent(dataloader: torch.utils.data.DataLoader, encoder: torch.nn.Module, device: str) -> list:
    encoded_latent = list()

    with torch.no_grad():
        encoder.eval()
        for batch_index, data in tqdm(enumerate(dataloader, 0)):
            batch_images = data[0]
            latent_vector = encoder(batch_images.to(device))
            encoded_latent.append(latent_vector)

    encoded_latent = torch.cat(encoded_latent).cpu()
    return encoded_latent


def eval_fid_score(fid_evaluator: FidEvaluator, generator: torch.nn.Module, device: str):
    return fid_evaluator.evaluate(generator, device) if fid_evaluator is not None else 0


def end_validation_epoch(outputs: dict, fid_evaluator: FidEvaluator, generator: torch.nn.Module, device: str):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    tensorboard_logs = {'val_loss': avg_loss}
    fid_score = eval_fid_score(fid_evaluator, generator, device)
    tensorboard_logs['fid_score'] = fid_score

    for k in outputs[0].keys():
        tensorboard_logs[k] = torch.stack([x[k] for x in outputs]).mean()

    print(tensorboard_logs)
    return {'avg_val_loss': avg_loss, 'fid_score': fid_score, 'log': tensorboard_logs}
