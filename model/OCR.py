import torch.nn as nn

from model.backbone import VGG
from model.seq2seq import Seq2Seq


class OCR(nn.Module):

    def __init__(self, vocab_size):
        super(OCR, self).__init__()
        self.cnn = VGG()
        self.transformer = Seq2Seq(vocab_size)

    def forward(self, img, tgt_input):
        emb = self.cnn(img)
        out = self.transformer(emb, tgt_input)
        return out