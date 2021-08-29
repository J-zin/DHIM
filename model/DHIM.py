import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModel, AutoTokenizer

from model.base_model import Base_Model
from utils.torch_helper import move_to_device, squeeze_dim

class DHIM(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def define_parameters(self):
        self.bert_model = AutoModel.from_pretrained('model/bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False 

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=c, kernel_size=(w, 768), padding=((w-1)//2, 0)) for (w, c) in [[1, 256], [3, 256], [5, 256]]])
        self.pro_layer = nn.Linear(768, self.hparams.encode_length)
        self.disc_g = nn.Linear(self.hparams.encode_length, self.hparams.encode_length)
        self.disc_l = nn.Linear(self.hparams.encode_length, self.hparams.encode_length)

    def forward(self, inputs):
        out_bert = self.bert_model(**inputs)[0]
        outputs  = torch.unsqueeze(out_bert[:, 1:], dim=1)
        con_x = [F.relu(torch.squeeze(conv(outputs), dim=-1)) for conv in self.convs]
        
        local_rep = torch.cat(con_x, dim=1).permute(0, 2, 1)
        global_rep = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in con_x], dim=1).permute(0, 2, 1)

        if self.hparams.median:
            code_l = self.pro_layer(local_rep)
            code_g = self.pro_layer(global_rep)
            code_x = self.pro_layer(out_bert[:, 0])
        else:
            prob_l = torch.sigmoid(self.pro_layer(local_rep) / self.hparams.alpha)
            prob_g = torch.sigmoid(self.pro_layer(global_rep) / self.hparams.alpha)
            prob_x = torch.sigmoid(self.pro_layer(out_bert[:, 0]) / self.hparams.alpha)

            code_l = hash_layer(prob_l - torch.rand_like(prob_l))
            code_g = hash_layer(prob_g - torch.rand_like(prob_g))
            code_x = hash_layer(prob_x - torch.rand_like(prob_x))

        code_l = code_l.view(-1, self.hparams.encode_length)
        code_g = code_g.view(-1, self.hparams.encode_length)

        ###################### local/global mutual information ###################################
        sim = torch.sigmoid(self.disc_l(code_g) @ (code_l.T))

        N, L = sim.shape[0], self.hparams.max_length
        mask = torch.zeros((N, N*L), dtype=bool)
        for i in range(N):
            mask[i, i*L:(i+1)*L] = 1
        positive_samples = sim[mask == 1].view(N, -1)
        negative_samples = sim[mask == 0].view(N, N-1, -1)
        
        MI_local = torch.mean(-F.softplus(-positive_samples) - (1 / (N - 1)) * torch.sum(F.softplus(negative_samples), dim=1), dim=-1)

        ###################### semantic-preserving regularizer #################################
        sim = torch.sigmoid(self.disc_g(code_g) @ (code_x.T))

        mask = torch.zeros((N, N), dtype=bool).fill_diagonal_(1)
        positive_samples = sim[mask == 1].view(N, -1)
        negative_samples = sim[mask == 0].view(N, -1)

        MI_global = torch.mean(-F.softplus(-positive_samples) - (1 / (N - 1)) * torch.sum(F.softplus(negative_samples), dim=-1, keepdims=True), dim=-1)


        ###################### final objective #################################
        loss = -torch.mean(MI_local + MI_global)

        return {'loss': loss, 'MI_local': torch.mean(MI_local), 'MI_global': torch.mean(MI_global)}
    
    def encode_discrete(self, target_inputs):
        outputs  = torch.unsqueeze(self.bert_model(**target_inputs)[0][:, 1:], dim=1)
        con_x = [F.relu(torch.squeeze(conv(outputs), dim=-1)) for conv in self.convs]
        global_rep = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in con_x], dim=1).permute(0, 2, 1)

        prob_g = torch.sigmoid(self.pro_layer(global_rep) / self.hparams.alpha)
        code_g = torch.squeeze(hash_layer(prob_g - 0.5), dim=1)
        return code_g

    def encode_continuous(self, target_inputs):
        outputs  = torch.unsqueeze(self.bert_model(**target_inputs)[0][:, 1:], dim=1)
        con_x = [F.relu(torch.squeeze(conv(outputs), dim=-1)) for conv in self.convs]
        global_rep = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in con_x], dim=1).permute(0, 2, 1)

        code_g = self.pro_layer(global_rep)
        code_g = code_g.view(-1, self.hparams.encode_length)
        return code_g

    def get_median_threshold_binary_code(self, database_loader, eval_loader, device):
        def extract_data(loader):
            encoding_chunks = []
            label_chunks = []
            for (docs, labels) in loader:
                docs = squeeze_dim(move_to_device(docs, device), dim=1)
                encoding_chunks.append(docs if self.encode_continuous is None else
                                   self.encode_continuous(docs))
                label_chunks.append(labels)

            encoding_mat = torch.cat(encoding_chunks, 0)
            label_mat = torch.cat(label_chunks, 0)
            label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
            return encoding_mat, label_lists

        train_rep, train_label = extract_data(database_loader)
        test_rep, test_label = extract_data(eval_loader)

        mid_val, _ = torch.median(train_rep, dim=0)

        train_b = (train_rep > mid_val).type(torch.FloatTensor).to(device)
        test_b = (test_rep > mid_val).type(torch.FloatTensor).to(device)

        del train_rep
        del test_rep
        return train_b, test_b, train_label, test_label

    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.parameters()}], lr = self.hparams.lr)

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'conv_out_dim': [50, 100, 200, 300, 400]
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-a", "--alpha", default = 1, type = float,
                            help = "Temperature for sigmoid function [%(default)d]",)
        parser.add_argument("-b", "--beta", default = 1, type = float,
                            help = "Temperature for MI regularizer [%(default)d]",)
        parser.add_argument("--conv_out_dim", default = 100, type = int,
                            help = "Dimention of the output of CNN [%(default)d]",)
        return parser

class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (torch.sign(input) + 1) // 2

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)