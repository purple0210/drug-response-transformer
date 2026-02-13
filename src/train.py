import torch
from src.multimodal_model import MultimodalModel

def train():
    model = MultimodalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(5):
        drug = torch.rand(32, 10)
        rna = torch.rand(32, 10)
        y = torch.rand(32, 1)

        pred = model(drug, rna)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("✅ Model saved as model.pt")

if __name__ == "__main__":
    train()
