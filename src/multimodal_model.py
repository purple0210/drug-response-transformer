import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.drug_encoder = nn.Linear(10, 16)
        self.rna_encoder = nn.Linear(10, 16)

        self.fc = nn.Linear(32, 1)

    def forward(self, drug, rna):
        drug_feat = self.drug_encoder(drug)
        rna_feat = self.rna_encoder(rna)

        combined = torch.cat([drug_feat, rna_feat], dim=1)
        return self.fc(combined)
