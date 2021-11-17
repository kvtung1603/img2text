import os
import time

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from Vocab import Vocab
from dataset.aug import ImgAugTransform
from dataset.dataloader import OCRdataset, ClusterRandomSampler, Collator
from loss.labelsmoothingloss import LabelSmoothingLoss
from model.OCR import OCR


class Train():
    def __init__(self, dataset_name, data_root, train_annotation, valid_annotation, data_aug = ImgAugTransform()):
        self.model = OCR()
        self.vocab = Vocab()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = SGD(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters)
        self.batch_size = 16
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        self.transforms = data_aug
        self.iter = 0

        self.dataset_name = dataset_name
        self.data_root = data_root
        self.train_annotation = train_annotation
        self.valid_annotation = valid_annotation
        self.train_losses = []

        self.train_gen = self.data_gen('train_{}'.format(self.dataset_name),
                                       self.data_root, self.train_annotation, self.masked_language_model,
                                       transform=self.transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen('valid_{}'.format(self.dataset_name),
                                           self.data_root, self.valid_annotation, masked_language_model=False)


    def make_dataloader(self, outputPath, root_dir, annotation_path, masked_language_model=True, transform=None):

        dataset = OCRdataset(outputPath, root_dir, annotation_path, self.vocab, transform=transform)
        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn, shuffle=True)

        return dataloader

    def train(self, ):

        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(
                    self.iter,
                    total_loss / self.print_every, self.optimizer.param_groups[0]['lr'],
                    total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                    self.iter, val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq


    def step(self, batch):

        self.model.train()
        self.model.to(self.device)

        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

        img = img.to(self.device)
        tgt_input = tgt_input.to(self.device)
        tgt_output = tgt_output.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)

        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)

        outputs = outputs.view(-1, outputs.size(2))#flatten(0, 1)
        tgt_output = tgt_output.view(-1)#flatten()

        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def save_weights(self, filename):
        os.makedirs(filename, exist_ok=True)

        torch.save(self.model.state_dict(), filename)

    def save_checkpoint(self, filename):
        state = {'iter': self.iter, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}

        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']
        self.train_losses = state_dict['train_losses']

