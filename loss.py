
import torch
import torch.nn as nn
import numpy as np
from codice import (
    carica_e_preprocesa,
    prepara_dati,
    crea_modello,
    train,
    valuta,
    plot_loss,
    plot_risultati,
)
 
# ============================================================
# ESPERIMENTO WEIGHTED — loss pesata per voti alti (>=80)
# ============================================================

def analisi_wheight_loss():
    # 1. Carica e preprocessa il dataset
    df = carica_e_preprocesa()
    
    # 2. Prepara i tensori
    X_train, X_test, y_train, y_test, y_train_raw = prepara_dati(df)
    
    # 3. Calcola i pesi: voti >= 80 pesano 3x
    weights        = np.where(y_train_raw.values >= 80, 2.0, 1.0)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    
    print(f"\nCampioni normali (< 80):  {(y_train_raw < 80).sum()}")
    print(f"Campioni pesati (>= 80):  {(y_train_raw >= 80).sum()}")
    
    # 4. Crea il modello
    model, optimizer = crea_modello(input_size=X_train.shape[1], lr=0.01)
    
    # 5. Loss con reduction='none' per applicare i pesi manualmente
    criterion = nn.MSELoss(reduction='none')
    
    # 6. Training con pesi
    losses = train(
        model, optimizer, criterion,
        X_train, y_train,
        epochs=200,
        weights_tensor=weights_tensor,
        label="WEIGHTED"
    )
    
    # 7. Grafico loss
    plot_loss(losses, label="WEIGHTED")
    
    # 8. Valutazione
    y_true, y_pred , mse, mae, r2, rmse= valuta(model, X_test, y_test, label="WEIGHTED")
    
    # 9. Grafici risultati
    plot_risultati(y_true, y_pred, label="WEIGHTED")

    print (mse, mae , r2  , rmse)
    return(mse, mae , r2  , rmse)