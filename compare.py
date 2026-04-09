from standard import (analisi_standard) 
from loss import (analisi_wheight_loss) 


A = analisi_standard()
print (A)

B = analisi_wheight_loss()
print (B)

diff_mse = A[0] - B[0]
diff_mae = A[1] - B[1]
diff_r2 = A[2] - B[2]
diff_rmse = A[3] - B[3]


print(f"{'Metrica':<10} {'Standard':>12} {'Weighted':>12} {'Differenza':>12}")
print("-" * 48)
print(f"{'MSE':<10} {A[0]:>12.4f} {B[0]:>12.4f} {A[0]-B[0]:>+12.4f}")
print(f"{'MAE':<10} {A[1]:>12.4f} {B[1]:>12.4f} {A[1]-B[1]:>+12.4f}")
print(f"{'R2':<10} {A[2]:>12.4f} {B[2]:>12.4f} {A[2]-B[2]:>+12.4f}")
print(f"{'RMSE':<10} {A[3]:>12.4f} {B[3]:>12.4f} {A[3]-B[3]:>+12.4f}")

print(f"MSE difference: {diff_mse:.4f}")
print(f"MAE difference: {diff_mae:.4f}")
print(f"R² difference:  {diff_r2:.4f}")
print(f"RMSE difference:  {diff_rmse:.4f}")

    