import numpy as np
import torch
import torch.nn as nn
import torch.utils as _utils
import snntorch as snn
from snntorch import surrogate #Used to get the surrogate gradient function
from scipy.signal import welch #Used for feature extraction Power Spectral Density(PSD)
from scipy.io import loadmat #Only used to load .mat files, not needed if using h5py for .h5 files
import h5py #Used to load .h5 files, not needed if using scipy.io for .mat files
import scipy.io
from sklearn.preprocessing import StandardScaler #Helps with normalization
from sklearn.model_selection import train_test_split #Split data into training and validation sets
from torch.utils.data import DataLoader, TensorDataset #Turns tensors into datasets and dataloaders for training
import matplotlib.pyplot as plt #Classing visualization library

#Hyperparameters
T = 100  # timesteps for rate encoding
BATCH_SIZE = 64
n_hidden1=128
n_hidden2=64
n_classes=2,
beta=0.9,
dropout_p=0.3
N_EPOCHS = 50
LR       = 1e-4

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch version: {torch.__version__}")
print(f"SNNTorch version: {snn.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}") 
'''
#Data loader functions to handle both legacy .mat and v7.3 HDF5-based .mat files
def safe_load_mat(file_path):
    """Handles both legacy .mat and v7.3 HDF5-based .mat files."""
    #Two matlab file formats: legacy (pre-7.3) and HDF5-based (7.3+). Try loading with scipy first, then fallback to h5py if needed.
    try:
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print(f"Loaded with scipy.io.loadmat.")
        return _clean_dict(data)
    except NotImplementedError:
        print(f"v7.3 file detected. Switching to h5py...")
        with h5py.File(file_path, 'r') as f:
            return {k: np.array(v) for k, v in f.items()}
    except Exception as e:
        print(f"Load failed: {e}")
        return None

def _clean_dict(d):#Converts MATLAB structs to nested Python dicts
    for key in d:
        if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):#Recursively converts leftover MATLAB structs from clean_dict into dicts
    dict_out = {}
    for strname in matobj._fieldnames:
        elem = getattr(matobj, strname)
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict_out[strname] = _todict(elem)
        else:
            dict_out[strname] = elem
    return dict_out

# --- Load Data ---
clean_data   = safe_load_mat('dataset.mat')
class_data   = safe_load_mat('class_012.mat')
rating_data  = safe_load_mat('rating.mat')

eeg_matrix   = clean_data['dataset']
eeg_matrix   = np.transpose(eeg_matrix, (2, 0, 1))  # → (45, 14, 19200)

subject_labels = class_data['class_012'].squeeze().astype(int)  # (45,) values 0,1,2
ratings        = rating_data['rating'].squeeze().astype(int)    # (45,) values 1-9

print(f"EEG matrix shape:    {eeg_matrix.shape}")
print(f"Subject labels:      {subject_labels}")
print(f"Ratings:             {ratings}")
print(f"Class distribution:  {np.bincount(subject_labels)}")

def extract_band_power(eeg_epoch, fs=128):
    """
    Compute band power for each channel across 4 EEG frequency bands.
    
    eeg_epoch: (n_channels, n_samples)
    returns:   (n_channels * 4,) feature vector
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30)
    }
    
    features = []
    n_channels = eeg_epoch.shape[0]
    
    for ch in range(n_channels):
        # nperseg=fs*2 = 2-second segments for Welch — good frequency resolution
        freqs, psd = welch(eeg_epoch[ch], fs=fs, nperseg=fs * 2)
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    
    return np.array(features)


def create_epochs_and_features(eeg_matrix, fs=128, epoch_len_sec=2, overlap=0.5):
    """
    Slice EEG into overlapping windows and extract features from each.
    
    overlap=0.5 means 50% overlap — effectively doubles dataset size.
    Adjacent epochs will be correlated, which is acceptable here.
    """
    epoch_samples = int(epoch_len_sec * fs)           # 256 samples @ 128Hz
    step_samples  = int(epoch_samples * (1 - overlap)) # 128 samples
    
    all_features = []
    all_subject_ids = []
    
    for subj in range(eeg_matrix.shape[0]):
        eeg = eeg_matrix[subj]  # (channels, samples)
        n_samples = eeg.shape[1]
        start = 0
        while start + epoch_samples <= n_samples:
            epoch = eeg[:, start:start + epoch_samples]
            all_features.append(extract_band_power(epoch, fs=fs))
            all_subject_ids.append(subj)
            start += step_samples
    
    return np.array(all_features), np.array(all_subject_ids)


features, subject_ids = create_epochs_and_features(eeg_matrix)
print(f"Features shape: {features.shape}")
print(f"Total epochs: {len(features)} across {eeg_matrix.shape[0]} subjects")
#Expected: (n_epochs, n_features) where n_features = n_channels * 4 (bands) = 14*4=56
def create_labels_from_subjects(subject_ids, subject_labels):
    """
    Each epoch inherits the label of the subject it came from.
    We binarize: 0 → Low (0), 1 and 2 → High (1).
    subject_ids:     (n_epochs,) — which subject each epoch belongs to
    subject_labels:  (45,)      — 0/1/2 label per subject
    """
    epoch_labels = subject_labels[subject_ids]
    # Binarize: Low=0, High=1 (moderate + high both become 1)
    labels = (epoch_labels >= 1).astype(np.int64)
    
    counts = np.bincount(labels)
    print(f"Epoch class distribution → Class0 (Low): {counts[0]} | Class1 (High): {counts[1]}")
    
    plt.figure(figsize=(6, 3))
    plt.bar(['Low', 'High'], counts, color=['steelblue', 'coral'])
    plt.ylabel('Number of epochs')
    plt.title('Epoch distribution by workload class (2-class)')
    plt.tight_layout()
    plt.show()
    
    return labels

labels = create_labels_from_subjects(subject_ids, subject_labels)

from sklearn.preprocessing import MinMaxScaler

# Step 1: Log-transform BEFORE normalizing.
# Band power spans many orders of magnitude (e.g. 0.001 to 10000 µV²).
# Log compresses that into a manageable range and kills outlier dominance.
# +1e-10 guards against log(0) on any zero-power values.
# Z-score each subject independently to remove inter-subject baseline differences
features_subj_norm = np.zeros_like(features)
for subj in np.unique(subject_ids):
    mask = subject_ids == subj
    subj_feats = features[mask]
    features_subj_norm[mask] = (subj_feats - subj_feats.mean(axis=0)) / (subj_feats.std(axis=0) + 1e-8)

# Then log + MinMax as before
features_log = np.log(np.abs(features_subj_norm) + 1e-10)

# Step 2: Per-feature MinMax to [0.05, 0.95]
# Each of the 56 features independently uses the full spike range.
# Slight inset from 0/1 edges — always-silent / always-firing neurons
# carry no information.
scaler = MinMaxScaler(feature_range=(0.05, 0.95))
features_norm = scaler.fit_transform(features_log)

# Diagnostic
print(f"Features min:  {features_norm.min():.4f}")   # ~0.05
print(f"Features max:  {features_norm.max():.4f}")   # ~0.95
print(f"Features mean: {features_norm.mean():.4f}")  # should be ~0.4-0.6
print(f"% above 0.6:   {(features_norm > 0.6).mean():.3f}")
print(f"% below 0.4:   {(features_norm < 0.4).mean():.3f}")

# Quick visual check — distribution should look roughly bell-shaped now
plt.figure(figsize=(10, 3))
plt.hist(features_norm.flatten(), bins=100, color='steelblue', alpha=0.8)
plt.xlabel('Normalized value')
plt.ylabel('Count')
plt.title('Feature distribution after log + MinMax normalization')
plt.tight_layout()
plt.show()

# --- Rate encode ---
# This was missing from the notebook. Do it here before the split
# so we only encode once, but index into it by subject after.
features_norm_tensor = torch.FloatTensor(features_norm)
spike_data = (torch.rand(T, len(features_norm), features_norm.shape[1]) < features_norm_tensor).float()
# shape: (T, n_epochs, n_features)

# --- Subject-based split (fixes label leakage) ---
all_subjects = np.unique(subject_ids)  # [0..44]
n_val_subjects = int(len(all_subjects) * 0.2)  # hold out ~9 subjects

rng = np.random.default_rng(42)
val_subjects = rng.choice(all_subjects, size=n_val_subjects, replace=False)
train_subjects = np.setdiff1d(all_subjects, val_subjects)

train_mask = np.isin(subject_ids, train_subjects)
val_mask   = np.isin(subject_ids, val_subjects)

# Permute: (T, n_epochs, F) → (n_epochs, T, F) for DataLoader
spike_data_batched = spike_data.permute(1, 0, 2)
labels_tensor = torch.LongTensor(labels)

train_dataset = TensorDataset(spike_data_batched[train_mask], labels_tensor[train_mask])
val_dataset   = TensorDataset(spike_data_batched[val_mask],   labels_tensor[val_mask])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)



print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")
print(f"Train epochs:   {train_mask.sum()} | Val epochs: {val_mask.sum()}")
print(f"Train class dist: {np.bincount(labels[train_mask])}")
print(f"Val class dist:   {np.bincount(labels[val_mask])}")


class EEG_SNN(nn.Module):
    def __init__(self, n_features, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_classes=n_classes,
                 beta=beta, T=T, dropout_p=dropout_p):
        super().__init__()
        self.T = T

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Layer 1
        self.fc1  = nn.Linear(n_features, n_hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout1 = nn.Dropout(p=dropout_p)

        # Layer 2
        self.fc2  = nn.Linear(n_hidden1, n_hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.dropout2 = nn.Dropout(p=dropout_p)

        # Output layer
        self.fc_out  = nn.Linear(n_hidden2, n_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (T, batch, features)

        mem1    = self.lif1.init_leaky()
        mem2    = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spike_out_acc = 0

        for t in range(self.T):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)

            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spike_out_acc = spike_out_acc + spk_out

        return spike_out_acc
    

snn_model = EEG_SNN(n_features = 56, n_classes=2, T=T).to(DEVICE)

total_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

snn_model = EEG_SNN(n_features = 56, n_classes=2, T=T).to(DEVICE)
optimizer = torch.optim.Adam(snn_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
class_counts = np.bincount(labels[train_mask])
class_weights = torch.FloatTensor(1.0 / np.sqrt(class_counts)).to(DEVICE)
class_weights = class_weights / class_weights.sum()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


train_losses, val_accs = [], []

for epoch in range(N_EPOCHS):
    snn_model.train()
    epoch_loss = 0
    
    for spike_batch, label_batch in train_loader:
        spike_batch = spike_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        
        optimizer.zero_grad()
        spike_counts = snn_model(spike_batch)
        loss = loss_fn(spike_counts, label_batch)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(snn_model.parameters(), 1.0)
        epoch_loss += loss.item()
    
    snn_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for spike_batch, label_batch in val_loader:
            spike_batch = spike_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            spike_counts = snn_model(spike_batch)
            preds = spike_counts.argmax(dim=1)
            correct += (preds == label_batch).sum().item()
            total   += len(label_batch)
    
    avg_loss = epoch_loss / len(train_loader)
    val_acc  = correct / total
    train_losses.append(avg_loss)
    val_accs.append(val_acc)
    scheduler.step(val_acc)
    
    #if (epoch + 1) % 5 == 0:
    print(f"Epoch {epoch+1:03d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f}")

print(f"\nBest Val Accuracy: {max(val_accs):.3f} (epoch {np.argmax(val_accs)+1})")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(train_losses, color='steelblue', linewidth=2)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('CrossEntropy Loss')
ax1.grid(alpha=0.3)

ax2.plot(val_accs, color='darkorange', linewidth=2)
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random chance (50%)')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('SNN Training Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Final val accuracy: {val_accs[-1]:.3f}")
print(f"Best val accuracy:  {max(val_accs):.3f} at epoch {np.argmax(val_accs)+1}")

from sklearn.metrics import confusion_matrix, classification_report
import itertools

snn_model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for spike_batch, label_batch in val_loader:
        spike_batch = spike_batch.to(DEVICE)
        spike_counts = snn_model(spike_batch)
        preds = spike_counts.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label_batch.numpy())

cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

class_names = ['Low', 'High']

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(2)); ax.set_xticklabels(class_names)
ax.set_yticks(range(2)); ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (row-normalized)')

for i, j in itertools.product(range(2), range(2)):
    ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
            ha='center', va='center',
            color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=9)

plt.tight_layout()
plt.show()

print(classification_report(all_labels, all_preds, target_names=class_names))
'''