#extract_scaler_from_keras.py
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from constants import ASV_MODEL_DIR, ASV_SCALER

# caută model best_model.keras în folderul ASV_MODEL_DIR
mfile = Path(ASV_MODEL_DIR) / "best_model.keras"
if not mfile.exists():
    raise SystemExit(f"Nu am găsit {mfile} — verifică {ASV_MODEL_DIR}")

model = tf.keras.models.load_model(str(mfile), compile=False)
print("Model loaded. Layers:")
for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))

# caută layer cu numele 'norm' (în train_cnn1d se numește 'norm')
norm_layer = None
for layer in model.layers:
    if getattr(layer, "name", "").lower().startswith("norm") or layer.__class__.__name__.lower().find("normalization")!=-1:
        norm_layer = layer
        break

if norm_layer is None:
    raise SystemExit("Normalization layer not found in model. Inspect model.layers above.")

# Keras Normalization memiliki atribut 'mean' dan 'variance' setelah adapt
mean = None
std = None
# Varianta TF2: layer.get_weights() -> [mean, variance, ...] (posibil diferit)
w = norm_layer.get_weights()
print("weights len:", len(w))
if len(w) >= 2:
    mean = np.array(w[0]).ravel()
    var  = np.array(w[1]).ravel()
    std = np.sqrt(np.maximum(var, 1e-12))
else:
    # alte implementări: layer.mean.numpy()
    try:
        mean = norm_layer.mean.numpy().ravel()
        var  = norm_layer.variance.numpy().ravel()
        std = np.sqrt(np.maximum(var, 1e-12))
    except Exception as e:
        raise SystemExit(f"Nu am putut extrage mean/var din Normalization layer: {e}")

print("mean.shape", mean.shape, "std.shape", std.shape)

# creează un StandardScaler echivalent și salvează-l
sc = StandardScaler()
sc.mean_ = mean
sc.var_ = var
sc.scale_ = std
sc.n_features_in_ = mean.shape[0]

outp = Path(ASV_SCALER)
outp.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(sc, str(outp))
print("Saved scaler to:", outp)
