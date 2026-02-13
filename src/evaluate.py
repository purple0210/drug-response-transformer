import torch
from src.multimodal_model import MultimodalModel

def evaluate():
    model = MultimodalModel()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    drug = torch.rand(1, 10)
    rna = torch.rand(1, 10)

    with torch.no_grad():
        pred = model(drug, rna)

    print("Predicted Response:", pred.item())

if __name__ == "__main__":
    evaluate()
