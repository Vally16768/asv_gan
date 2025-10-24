# === Dataset location (ASVspoof 2019 LA) ===
# If ASVSPOOF_ROOT is not set, fall back to "dataset"
directory = "database/data/asvspoof2019"

# === Dataset / split settings ===
sampling_rate   = 16000
random_state    = 42

# Folosim LA_dev ca TEST (oficial), deci nu mai decupăm "test intern" din train.
# Adăugăm mărimea de VALIDARE (din LA_train) — stratificat pe etichetă și grupat pe speaker.
validation_size = 0.10

# (Optional / legacy) test_size rămâne nefolosit când folosim LA_dev ca test.
# Îl păstrăm doar ca să nu rupem scripturi vechi care îl citesc.
test_size = 0.20

# === Folders / filenames ===
index_folder_name                       = "index"
results_folder                          = "results"
temp_data_folder_name                   = "temp_data"
models_folder                           = "models"

save_evaluation_model_results_file_name = "eval_results.json"
save_the_best_combination_file_name     = "combinations_ordered_by_accuracy.txt"
save_combinations_file_name             = "combinations_accuracy.txt"

# Common artifact filenames for training runs
train_log_filename              = "train_log.csv"
best_model_filename             = "best_model.keras"
final_model_filename            = "final_model.keras"
accuracy_txt_filename           = "accuracy.txt"
classification_report_filename  = "classification_report.txt"
predictions_csv_filename        = "predictions.csv"
confusion_matrix_png_filename   = "confusion_matrix.png"

# === Feature name mappings (for combos/tags, etc.) ===
feature_name_mapping = {
    "mfcc": "A",
    "mfcc_delta": "B",
    "centroid": "C",
    "bandwidth": "D",
    "contrast": "E",
    "rolloff": "F",
    "flatness": "G",
    "chroma": "H",
    "rms": "I",
    "zcr": "J",
}
feature_name_reverse_mapping = {v: k for k, v in feature_name_mapping.items()}

# === Default run “combo” name for the 1D-CNN MFCC model ===
cnn1d_default_combo_name = f"{feature_name_mapping['mfcc']}_mfcc128"  # "A_mfcc128"

# === 1D-CNN (MFCC) training defaults ===
window_length_ms     = 10
fmax                 = 8000
n_mels                 = 128
n_mfcc           = 128
cnn1d_duration_seconds = 4.0
cnn1d_batch_size       = 64
cnn1d_epochs           = 200
cnn1d_seed             = random_state

# Model / optimizer hyperparameters
cnn1d_dropout1         = 0.10
cnn1d_dropout2         = 0.25
cnn1d_optimizer_lr     = 1e-3

# Callback settings
cb_model_checkpoint_monitor = "val_loss"

cb_early_stopping_monitor  = "val_loss"
cb_early_stopping_patience = 10
cb_early_stopping_restore  = True

cb_reduce_lr_monitor = "val_loss"
cb_reduce_lr_factor  = 0.5
cb_reduce_lr_patience= 3
cb_reduce_lr_min_lr  = 1e-6


# ASVspoof 2019 LA subdirectories (relative to `directory`)
la_train_flac_subdir = "ASVspoof2019_LA_train/flac"
la_dev_flac_subdir   = "ASVspoof2019_LA_dev/flac"
la_eval_flac_subdir  = "ASVspoof2019_LA_eval/flac"
la_protocols_subdir  = "ASVspoof2019_LA_cm_protocols"

# Protocol filenames (inside la_protocols_subdir)
la_train_trn_filename = "ASVspoof2019.LA.cm.train.trn.txt"
la_dev_trl_filename   = "ASVspoof2019.LA.cm.dev.trl.txt"
la_eval_trl_filename  = "ASVspoof2019.LA.cm.eval.trl.txt"
