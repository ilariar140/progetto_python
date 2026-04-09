import os
import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


torch.manual_seed(42)
np.random.seed(42)

DATASET_ID = "grandmaster07/student-exam-performance-dataset-analysis"
FILE_NAME  = "StudentPerformanceFactors.csv"
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "student")


# ──────────────────────────────────────────────
# 1. CARICA E PREPROCESSA
# ──────────────────────────────────────────────

def carica_e_preprocesa():
    # crea la cartella
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # scarica dataset
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    print("Dataset salvato in:", dataset_path)

    # costruisci il path corretto al file
    file_path = os.path.join(dataset_path, FILE_NAME)

    # leggi il csv
    df = pd.read_csv(file_path)
    print(df.head())

    # contare i nulli
    print(df.isnull().sum())
    rows_to_drop = []

    for i, row in df.iterrows():
        # Controlla se c'è almeno un valore nullo o vuoto
        if any(value == '' or pd.isna(value) for value in row):
            rows_to_drop.append(i)

    # Rimuovi le righe
    df = df.drop(rows_to_drop)
    print(df.isnull().sum())

    # Trasformazione da categorici a numerici
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Internet_Access'] = le.fit_transform(df['Internet_Access'])
    df['School_Type'] = le.fit_transform(df['School_Type'])
    df['Extracurricular_Activities'] = le.fit_transform(df['Extracurricular_Activities'])
    df['Learning_Disabilities'] = le.fit_transform(df['Learning_Disabilities'])
    df['Peer_Influence'] = le.fit_transform(df['Peer_Influence'])

    cols = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
            'Family_Income', 'Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
    enc = OrdinalEncoder(dtype=int)
    df[cols] = enc.fit_transform(df[cols])
    print(df.head())

    return df


# ──────────────────────────────────────────────
# 2. PREPARA DATI
# ──────────────────────────────────────────────

def prepara_dati(df):
    A = "Exam_Score"

    x = df.drop(columns=[A])
    y = df[A]

    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_train


# ──────────────────────────────────────────────
# RETI NEURALI
# ──────────────────────────────────────────────

class ExamScoreNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Hidden layer 1
        self.fc1 = nn.Linear(input_size, 16)
        # Hidden layer 2
        self.fc2 = nn.Linear(16, 8)
        # Output layer (1 neurone lineare)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # attivazione ReLU
        x = torch.relu(self.fc2(x))  # attivazione ReLU
        x = self.out(x)              # output lineare
        return x


# ──────────────────────────────────────────────
# 3. CREA MODELLO
# ──────────────────────────────────────────────

def crea_modello(input_size, lr=0.01):
    model = ExamScoreNN(input_size)

    # stampa parametri
    for nome, param in model.named_parameters():
        print(f"{nome:15s} shape={param.shape}")

    totale = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {totale}")

    # Ottimizzatore Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(model)

    return model, optimizer


# ──────────────────────────────────────────────
# 4. TRAIN
# ──────────────────────────────────────────────

def train(model, optimizer, criterion, X_train_tensor, y_train_tensor,weights_tensor=None, epochs=200, label=""):
    train_losses = []

    print(f"Inizio addestramento {label}...")

    for epoch in range(epochs):
        # Reset dei gradienti (fondamentale per PyTorch)
        optimizer.zero_grad()
        # Forward pass: predizione del modello
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor) 

        if weights_tensor is not None:
            loss = (loss * weights_tensor).mean()

        # Backpropagation: calcolo del gradiente
        loss.backward()

        # Aggiorna i pesi con l'optimizer
        optimizer.step()

        # Salviamo la loss per il grafico
        train_losses.append(loss.item())

        # Controlla la loss ogni 20 epoche
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print(f"Addestramento {label} completato!")

    return train_losses


# ──────────────────────────────────────────────
# 5. VALUTA
# ──────────────────────────────────────────────

def valuta(model, X_test_tensor, y_test_tensor, label=""):
    criterion = nn.MSELoss()

    # Passiamo il modello in modalità valutazione
    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss   = criterion(predictions, y_test_tensor)

        # Calcoliamo anche il Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(predictions - y_test_tensor))

        print("-" * 30)
        print(f"MSE sul Test Set: {test_loss.item():.4f}")
        print(f"Errore medio (MAE) sul punteggio dell'esame: {mae.item():.2f} punti")

    # Predizione sul test set
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    # Convertiamo i tensori in array NumPy
    y_test_numpy = y_test_tensor.numpy()
    y_pred_numpy = y_pred_tensor.numpy()

    # Calcolo delle metriche di regressione
    mse  = mean_squared_error(y_test_numpy, y_pred_numpy)
    mae  = mean_absolute_error(y_test_numpy, y_pred_numpy)
    r2   = r2_score(y_test_numpy, y_pred_numpy)
    rmse = np.sqrt(mean_squared_error(y_test_numpy, y_pred_numpy))

    print(f"--- METRICHE DI VALUTAZIONE {label} ---")
    print(f"MSE (Mean Squared Error): {mse:.4f}")   # penalizza errori grandi
    print(f"MAE (Mean Absolute Error): {mae:.4f}")  # errore medio
    print(f"R² Score: {r2:.4f}")                    # 1 - 0
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)

    return y_test_numpy, y_pred_numpy, mse, mae, r2, rmse


# ──────────────────────────────────────────────
# 6. PLOT LOSS
# ──────────────────────────────────────────────

def plot_loss(train_losses, label=""):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Training Loss (MSE) {label}')
    plt.title('Andamento della Loss durante il Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# ──────────────────────────────────────────────
# 7. PLOT RISULTATI
# ──────────────────────────────────────────────

def plot_risultati(y_test_numpy, y_pred_numpy, label=""):
    plt.figure(figsize=(12, 5))

    # Grafico 1: Scatter plot (Valori Reali vs Predetti)
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_numpy, y_pred_numpy, alpha=0.5, color='teal')
    plt.plot([y_test_numpy.min(), y_test_numpy.max()],
             [y_test_numpy.min(), y_test_numpy.max()],
             'r--', lw=2)  # Linea di perfezione
    plt.title(f'Valori Reali vs Predizioni {label}')
    plt.xlabel('Voti Reali')
    plt.ylabel('Voti Predetti')
    plt.grid(True)

    # Grafico 2: Distribuzione degli Errori (Residui)
    plt.subplot(1, 2, 2)
    errors = y_test_numpy - y_pred_numpy
    plt.hist(errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='blue', linestyle='--')
    plt.title(f'Distribuzione degli Errori {label}')
    plt.xlabel('Errore (Reale - Predetto)')
    plt.ylabel('Frequenza')
    plt.grid(True)

    plt.tight_layout()
    plt.show()