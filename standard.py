
import torch.nn as nn
from codice import (
    carica_e_preprocesa,
    prepara_dati,
    crea_modello,
    train,
    valuta,
    plot_loss,
    plot_risultati,
)

# ESPERIMENTO STANDARD — loss normale
# ============================================================
def analisi_standard():
# 1. Carica e preprocessa il dataset
    df = carica_e_preprocesa()
    
    # 2. Prepara i tensori
    X_train, X_test, y_train, y_test, y_train_raw = prepara_dati(df)
    
    # 3. Crea il modello
    model, optimizer = crea_modello(input_size=X_train.shape[1], lr=0.01)
    
    # 4. Loss standard
    criterion = nn.MSELoss()
    
    # 5. Training
    losses = train(
        model, optimizer, criterion,
        X_train, y_train,
        epochs=200,
        label="STANDARD"
    )
    
    # 6. Grafici loss
    plot_loss(losses, label="STANDARD")
    
    # 7. Valutazione
    y_true, y_pred, mse, mae, r2, rmse = valuta(model, X_test, y_test, label="STANDARD")
    
    # 8. Grafici risultati
    plot_risultati(y_true, y_pred, label="STANDARD")
    
    print (mse, mae , r2  , rmse)
    return(mse, mae , r2  , rmse)