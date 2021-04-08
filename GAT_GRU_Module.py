#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class GAT_GRU():

    def __init__(self, multiheadargs):

        super(GAT_GRU, self).__init__()

        self.mtl1 = None
        self.mtl2 = None

    def forward(X, h, A):
        '''
        input:
            X: Batch x N_Nodes * InputDim
            h: Batch x N_Nodes * InputDim
            A: N_Nodes x N_Nodes
        '''
        ## TODO:
        ## multihead GA: how to use adjacency mat

        ## TODO: hid1 -> rt
        hid1 = torch.cat([X, h], dim=-1)
        g1, k1 = self.mtl1(hid1, A)
        hid1 = torch.cat([g1, k1], dim=-1)
        hid1 = F.sigmoid(hid1)

        ## TODO: how [gt, kt] and it co-exist?
        ## TODO: hid2 -> bt
        hid2 = torch.cat([X, torch.dot(hid1, h)], dim=-1)
        it = self.mtl2(hid2, X)
        hid2 = F.tanh(it)
        
        ut = torch.cat([g1, k1], dim=-1)
        out = torch.dot(ut, bt) + torch.dot(1 - ut, h)

        return out

## test case:
gatgru = GAT_GRU({})

