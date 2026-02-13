🚀 How to Run This Project Locally
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/drug-response-transformer.git
cd drug-response-transformer

2️⃣ Create a Conda Environment (Recommended)
conda create -n drug_response python=3.9 -y
conda activate drug_response

3️⃣ Install Dependencies
pip install -r requirements.txt


If requirements.txt fails:

pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm fastapi uvicorn

4️⃣ Verify PyTorch Installation
python -c "import torch; print(torch.__version__)"

🧠 Training the Model
python -m src.train


This will:

Load dataset

Train the multimodal transformer

Save the trained model as model.pt

📊 Evaluate the Model
python -m src.evaluate

🔍 Interpret Attention
python -m src.interpret_attention

🚀 Running the API (Local Deployment)
1️⃣ Start the Server
uvicorn deployment.app:app --reload


Server will run at:

http://127.0.0.1:8000

2️⃣ Open Interactive API Docs

Visit:

http://127.0.0.1:8000/docs

3️⃣ Example Prediction Request
{
  "drug": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
  "rna":  [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
}

📂 Project Structure
DRUG-RESPONSE-TRANSFORMER/
│
├── data/                  # Dataset files
├── deployment/            # FastAPI deployment files
├── scripts/               # Utility scripts
├── src/                   # Model and training code
│   ├── dataset.py
│   ├── drug_encoder.py
│   ├── rna_encoder.py
│   ├── multimodal_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── interpret_attention.py
│
├── model.pt               # Trained model (generated after training)
├── requirements.txt
└── README.md

⚠️ Common Issues
❌ ModuleNotFoundError: src

Run using:

python -m src.train


instead of:

python src/train.py

❌ CUDA Errors

If you don’t have GPU:

torch.load("model.pt", map_location="cpu")

🖥️ System Requirements

Python 3.9+

Conda (recommended)

PyTorch

8GB RAM minimum

Optional: NVIDIA GPU (for faster training)
