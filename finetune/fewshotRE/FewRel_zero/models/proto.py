import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AutoModel


class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, pretrained_path, path, hidden_size=230, alpha=0.5):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.bert = nn.DataParallel(AutoModel.from_pretrained(pretrained_path).cuda())
        state_dict = torch.load(path)
        if 'rbert' in state_dict:
            self.bert.module.load_state_dict(state_dict['rbert'], strict=False)
        self.alpha = alpha

    def __dist__(self, x, y, dim):
        #return (torch.pow(x - y, 2)).sum(dim)
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, desc, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        s_cls, support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        q_cls, query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

        hidden_size = support.size(-1)
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size) # (B, total_Q, D)

        B = support.size(0) # Batch size

        # s_cls = self.drop(s_cls)
        q_cls = self.drop(q_cls)

        # s_cls = s_cls.view(B, N, K, -1)
        q_cls = q_cls.view(B, total_Q, -1)

        # s_cls = torch.mean(s_cls, 2)

        r_cls = self.bert(**desc)[1].view(B, N, -1)
        rq_logits = self.__batch_dist__(r_cls, q_cls)  # (B, total_Q, N)

        # rs_logits = (s_cls.view(B, N, 1, -1) * r_cls.view(B, 1, N, -1)).sum(-1)  # (B, N, N)


        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        # minn, _ = logits.min(-1)
        # logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        # _, pred = torch.max(logits.view(-1, N+1), 1)

        N = logits.size(-1)
        # rq_p = torch.softmax(rq_logits, dim=2)
        # sq_p = torch.softmax(logits, dim=2)
        p = self.alpha * logits + (1.0 - self.alpha) * rq_logits
        p = torch.log_softmax(p, dim=2)
        _, pred = torch.max(p.view(-1, N), 1)
        return p, pred

    
    
    
