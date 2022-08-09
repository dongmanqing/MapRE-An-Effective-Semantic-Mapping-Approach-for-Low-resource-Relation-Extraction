import torch
import pdb 
import torch.nn as nn
from transformers import BertModel


class REModel(nn.Module):
    """relation extraction model
    """
    def __init__(self, args, load_rbert, weight=None):
        super(REModel, self).__init__()
        self.args = args 
        self.training = True
        
        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        # scale = 2 if args.entity_marker else 1
        if args.model == 'cls':
            scale = 1
        elif args.model == 'ht':
            scale = 2
        elif args.model == 'htcls':
            scale = 3
        else:
            raise

        self.rel_fc = nn.Linear(args.hidden_size*scale, args.rel_num)
        # self.linear_map = nn.Sequential(
        #     nn.Linear(args.hidden_size*2, args.hidden_size)
        # )
        self.linear_map = nn.Linear(args.hidden_size*scale, args.hidden_size)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rbert = BertModel.from_pretrained('bert-base-uncased')
        if args.ckpt_to_load != "None":
            print("********* load from "+args.ckpt_to_load+" ***********")
            ckpt = torch.load(args.ckpt_to_load)
            self.bert.load_state_dict(ckpt["bert-base"], strict=False)
            if load_rbert:
                self.rbert.load_state_dict(ckpt["rbert"], strict=False)
        else:
            print("*******No ckpt to load, Let's use bert base!*******")
        
    def forward(self, rel, input_ids, mask, h_pos, t_pos, label):
        # bert encode
        outputs = self.bert(input_ids, mask)

        # entity marker
        if self.args.model == 'ht':
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1) #(batch_size, hidden_size*2)
        elif self.args.model == 'cls':
            #[CLS]
            state = outputs[0][:, 0, :] #(batch_size, hidden_size)
        else:
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1)
            state = torch.cat((state, outputs[0][:, 0, :]), dim=1)

        batch_size = state.size(0)
        # linear map
        logits = self.rel_fc(state) #(batch_size, rel_num)


        # rcls = self.rbert(**rel)[0][:, 0, :]
        # if self.args.freeze_rbert:
        #     rcls = rcls.detach()

        # logits = (self.linear_map(state).view(batch_size, 1, -1) * rcls.view(1, rcls.size(0), -1)).sum(dim=-1)

        # logits = 0.9 * logits + 0.1 * rqlogits

        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output    
        




