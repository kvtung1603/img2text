import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from Vocab import Vocab
from dataset.aug import ImgAugTransform
from dataset.dataloader import OCRdataset, ClusterRandomSampler, Collator
from dataset.logger import Logger
from loss.labelsmoothingloss import LabelSmoothingLoss
from metrics import compute_accuracy
from model.OCR import OCR
from utils import translate


class Train():
    def __init__(self, chars, dataset_name, data_root, train_annotation, valid_annotation, checkpoint, data_aug = ImgAugTransform()):
        self.vocab = Vocab(chars)
        self.model = OCR(len(self.vocab))

        self.num_iters = 20000
        self.print_every = 200
        self.valid_every = 3000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.001, total_steps=self.num_iters, pct_start=0.1)
        self.batch_size = 16
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        self.transforms = data_aug
        self.iter = 0

        self.dataset_name = dataset_name
        self.data_root = data_root
        self.train_annotation = train_annotation
        self.valid_annotation = valid_annotation

        self.train_losses = []
        self.checkpoint = checkpoint
        self.metrics = 10000
        self.masked_language_model = True
        self.logger = Logger("./train.log")

        self.train_gen = self.make_dataloader('train_{}'.format(self.dataset_name),
                                       self.data_root, self.train_annotation, self.masked_language_model,
                                       transform=self.transforms)

        self.valid_gen = self.make_dataloader('valid_{}'.format(self.dataset_name),
                                        self.data_root, self.valid_annotation, masked_language_model=False)

    def make_dataloader(self, outputPath, root_dir, annotation_path, masked_language_model=True, transform=None):

        dataset = OCRdataset(outputPath, root_dir, annotation_path, self.vocab, transform=transform)
        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, sampler=sampler, num_workers=3, collate_fn=collate_fn, pin_memory=True)
        return dataloader

    def train(self):
        if self.checkpoint is not None:
            self.load_weights()

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
                    self.save_checkpoint(self.checkpoint)
                    best_acc = acc_full_seq

    def validate(self):
        self.model.eval()
        self.model.to(self.device)

        total_loss = []

        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                img = batch['img'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                tgt_padding_mask = batch['tgt_padding_mask'].to(self.device)

                output = self.model(img, tgt_input, tgt_padding_mask)

                output = output.flatten(0, 1)
                tgt_output = tgt_output.flatten()

                loss = self.criterion(output, tgt_output)

                total_loss.append(loss.item())

                del output
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []

        for batch in self.valid_gen:
            img = batch['img'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            tgt_padding_mask = batch['tgt_padding_mask'].to(self.device)

            translated_sentence, prob = translate(img, self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(tgt_output.tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None):
        pred_sents, actual_sents, _, _ = self.predict(sample=sample)
        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        return acc_full_seq, acc_per_char

    def step(self, batch):
        self.model.train()
        self.model.to(self.device)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

        img = img.to(self.device)
        tgt_input = tgt_input.to(self.device)
        tgt_output = tgt_output.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)

        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)

        outputs = outputs.view(-1, outputs.size(2)) #flatten(0, 1)
        tgt_output = tgt_output.view(-1) #flatten()

        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def save_checkpoint(self, filename):
        state = {'iter': self.iter, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        torch.save(state, filename)

    def load_weights(self):
        state_dict = torch.load(self.checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']
        self.train_losses = state_dict['train_losses']


    def train_v1(self):
        best_acc = 0.0
        self.model.train()
        for i in range(self.num_epochs):
            epoch_loss = 0.0
            start = time.time()
            for _, batch in enumerate(self.train_gen):

                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch[
                    'tgt_padding_mask']
                img = img.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_padding_mask = tgt_padding_mask.to(self.device)
                tgt_output = tgt_output.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(img, tgt_input, tgt_padding_mask)
                outputs = outputs.flatten(0,1)
                tgt_output = tgt_output.flatten()

                loss = self.criterion(outputs, tgt_output)
                epoch_loss += loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_time = time.time() - start
            info = "epoch: {:d} - train loss: {:.3f} - lr {:.2e} - train time: {:.2f}".format(i, epoch_loss, self.optimizer.param_groups[0]['lr'], train_time)

            print(info)
            self.logger.log(info)

            if i%5 == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)
                info = 'epoch: {:d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                    i, val_loss, acc_full_seq, acc_per_char)

                print(info)
                self.logger.log(info)
            if acc_full_seq > best_acc:
                best_acc = acc_full_seq
                filename = "checkpoint{:.0f}.pth".format(acc_full_seq)
                self.save_checkpoint(filename)




