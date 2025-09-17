"""
Audio Processing Module for Political Speech Applause Detection
Handles loading, preprocessing, and feature extraction from audio files
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """Process audio files and extract features for applause detection"""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([])
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        if len(audio) == 0:
            return np.zeros((1, self.n_mfcc))
        
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc
        )
        # Transpose to get (time_steps, n_mfcc) format
        return mfcc.T
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features (pitch, energy, etc.)"""
        if len(audio) == 0:
            return {
                'mean_pitch': 0.0,
                'max_pitch': 0.0,
                'min_pitch': 0.0,
                'pitch_range': 0.0,
                'mean_energy': 0.0,
                'max_energy': 0.0,
                'energy_std': 0.0,
                'zero_crossing_rate': 0.0
            }
        
        # Extract pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = pitches[pitches > 0]
        
        # Extract energy (RMS)
        rms = librosa.feature.rms(y=audio)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'mean_pitch': np.mean(pitch_values) if len(pitch_values) > 0 else 0.0,
            'max_pitch': np.max(pitch_values) if len(pitch_values) > 0 else 0.0,
            'min_pitch': np.min(pitch_values) if len(pitch_values) > 0 else 0.0,
            'pitch_range': (np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 0 else 0.0,
            'mean_energy': np.mean(rms),
            'max_energy': np.max(rms),
            'energy_std': np.std(rms),
            'zero_crossing_rate': np.mean(zcr)
        }
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract rhythm and tempo features"""
        if len(audio) == 0:
            return {
                'tempo': 0.0,
                'spectral_centroid_mean': 0.0,
                'spectral_rolloff_mean': 0.0,
                'spectral_bandwidth_mean': 0.0
            }
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        return {
            'tempo': tempo,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth)
        }
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all audio features"""
        mfcc = self.extract_mfcc_features(audio)
        spectral = self.extract_spectral_features(audio)
        rhythm = self.extract_rhythm_features(audio)
        
        # Combine all features
        features = {
            'mfcc': mfcc,
            **spectral,
            **rhythm
        }
        
        return features

class ApplauseDataset(Dataset):
    """PyTorch Dataset for applause detection"""
    
    def __init__(self, audio_files: List[str], labels: List[int], processor: AudioProcessor, max_length: int = 200):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and process audio
        audio = self.processor.load_audio(audio_file)
        features = self.processor.extract_all_features(audio)
        
        # Convert to tensors
        mfcc = features['mfcc']
        
        # Pad or truncate to max_length
        if mfcc.shape[0] > self.max_length:
            mfcc = mfcc[:self.max_length]
        else:
            # Pad with zeros
            padding = np.zeros((self.max_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack([mfcc, padding])
        
        mfcc_tensor = torch.FloatTensor(mfcc)
        
        # Create feature vector from other features
        other_features = [
            features['mean_pitch'], features['max_pitch'], features['min_pitch'],
            features['pitch_range'], features['mean_energy'], features['max_energy'],
            features['energy_std'], features['zero_crossing_rate'], features['tempo'],
            features['spectral_centroid_mean'], features['spectral_rolloff_mean'],
            features['spectral_bandwidth_mean']
        ]
        other_features_tensor = torch.FloatTensor(other_features)
        
        return {
            'mfcc': mfcc_tensor,
            'other_features': other_features_tensor,
            'label': torch.LongTensor([label])
        }

def create_data_loaders(audio_dirs: List[str], label_files: List[str], 
                       batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    processor = AudioProcessor()
    
    # Load all audio files and labels
    all_audio_files = []
    all_labels = []
    
    for audio_dir, label_file in zip(audio_dirs, label_files):
        if os.path.exists(audio_dir) and os.path.exists(label_file):
            # Get audio files
            audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                          if f.endswith('.wav')]
            
            # Load labels
            labels_df = pd.read_csv(label_file, header=None, names=['file', 'start', 'end'])
            
            # Create labels (1 for applause, 0 for non-applause)
            label_value = 1 if 'applause' in audio_dir else 0
            labels = [label_value] * len(audio_files)
            
            all_audio_files.extend(audio_files)
            all_labels.extend(labels)
    
    # Create dataset
    dataset = ApplauseDataset(all_audio_files, all_labels, processor)
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the audio processor
    processor = AudioProcessor()
    print("Audio processor initialized successfully!")
    
    # Test with a sample file if available
    sample_dirs = ['applause_pt1', 'applause_pt2', 'non_applause_pt1', 'non_applause_pt2']
    sample_labels = ['PennSound_applause_labels.csv', 'PennSound_applause_labels.csv', 
                    'PennSound_non_applause_labels.csv', 'PennSound_non_applause_labels.csv']
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(sample_dirs, sample_labels, batch_size=16)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch shapes: MFCC: {batch['mfcc'].shape}, Other features: {batch['other_features'].shape}")
        print(f"Labels: {batch['label'].shape}")
        break