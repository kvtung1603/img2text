import torch.nn as nn

from backbone import VGG
from seq2seq import Seq2Seq


class OCR(nn.Module):

    def __init__(self, vocab_size):
        self.cnn = VGG()
        self.transformer = Seq2Seq(vocab_size)

    def forward(self, img, tgt_input):
        emb = self.cnn(img)
        out = self.transformer(emb, tgt_input)
        return out