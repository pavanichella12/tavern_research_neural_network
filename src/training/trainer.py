"""
Training Pipeline for Applause Detection Model
Handles training, validation, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import time

class ApplauseTrainer:
    """Trainer class for applause detection model"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and loss function
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            mfcc = batch['mfcc'].to(self.device)
            other_features = batch['other_features'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(mfcc, other_features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                mfcc = batch['mfcc'].to(self.device)
                other_features = batch['other_features'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(mfcc, other_features)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return epoch_loss, epoch_acc, metrics
    
    def calculate_metrics(self, y_true: List, y_pred: List, y_prob: List) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['precision'] = report['weighted avg']['precision']
        metrics['recall'] = report['weighted avg']['recall']
        metrics['f1'] = report['weighted avg']['f1-score']
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              save_dir: str = 'models',
              early_stopping_patience: int = 10) -> Dict:
        """Main training loop"""
        
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📈 Training samples: {len(train_loader.dataset)}")
        print(f"📈 Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"📊 Val F1: {val_metrics['f1']:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save model
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_metrics': val_metrics,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print("💾 Best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time:.2f} seconds")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'total_epochs': epoch + 1
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        print("🔍 Evaluating model on test set...")
        
        val_loss, val_acc, metrics = self.validate_epoch(test_loader)
        
        print(f"📊 Test Loss: {val_loss:.4f}")
        print(f"📊 Test Accuracy: {val_acc:.2f}%")
        print(f"📊 Test F1-Score: {metrics['f1']:.4f}")
        print(f"📊 Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Print classification report
        print("\n📋 Classification Report:")
        print(classification_report(
            [0] * len(test_loader.dataset), [0] * len(test_loader.dataset),  # Placeholder
            target_names=['No Applause', 'Applause']
        ))
        
        return metrics

def load_model(model_path: str, model_class: nn.Module, device: str = 'cpu') -> nn.Module:
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"📊 Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model

if __name__ == "__main__":
    print("🎯 Applause Detection Trainer")
    print("This module provides training functionality for the applause detection model.")
    print("Import and use the ApplauseTrainer class to train your models!")