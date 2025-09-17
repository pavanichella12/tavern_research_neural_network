"""
Main Training Script for Political Speech Applause Detection
Trains a neural network to predict audience applause in political speeches
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing.audio_processor import create_data_loaders
from models.applause_model import create_model, count_parameters
from training.trainer import ApplauseTrainer

def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train applause detection model')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'transformer'], help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🎯 Political Speech Applause Detection Training")
    print("=" * 60)
    print(f"🖥️  Device: {device}")
    print(f"🏗️  Model: {args.model_type.upper()}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📅 Epochs: {args.epochs}")
    print(f"🎓 Learning rate: {args.learning_rate}")
    print(f"🧠 Hidden dimension: {args.hidden_dim}")
    print("=" * 60)
    
    # Check if data exists
    data_dirs = ['applause_pt1', 'applause_pt2', 'non_applause_pt1', 'non_applause_pt2']
    label_files = ['PennSound_applause_labels.csv', 'PennSound_applause_labels.csv',
                   'PennSound_non_applause_labels.csv', 'PennSound_non_applause_labels.csv']
    
    missing_dirs = [d for d in data_dirs if not os.path.exists(d)]
    missing_files = [f for f in label_files if not os.path.exists(f)]
    
    if missing_dirs or missing_files:
        print("❌ Missing data files:")
        for d in missing_dirs:
            print(f"   - {d}")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all data files are in the current directory.")
        return
    
    try:
        # Create data loaders
        print("\n📊 Loading and preprocessing data...")
        train_loader, val_loader = create_data_loaders(
            audio_dirs=data_dirs,
            label_files=label_files,
            batch_size=args.batch_size,
            train_split=0.8
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   📈 Training batches: {len(train_loader)}")
        print(f"   📈 Validation batches: {len(val_loader)}")
        
        # Create model
        print(f"\n🏗️  Creating {args.model_type.upper()} model...")
        model = create_model(
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            dropout=0.3
        )
        
        print(f"✅ Model created!")
        print(f"   📊 Parameters: {count_parameters(model):,}")
        
        # Create trainer
        print(f"\n🎓 Initializing trainer...")
        trainer = ApplauseTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate
        )
        
        print(f"✅ Trainer initialized!")
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Train model
        print(f"\n🚀 Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            early_stopping_patience=10
        )
        
        # Plot training history
        print(f"\n📊 Plotting training history...")
        trainer.plot_training_history(os.path.join(args.save_dir, 'training_history.png'))
        
        # Final evaluation
        print(f"\n🔍 Final evaluation...")
        final_metrics = trainer.evaluate_model(val_loader)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best validation loss: {history['best_val_loss']:.4f}")
        print(f"⏱️  Total training time: {history['training_time']:.2f} seconds")
        print(f"📁 Model saved to: {args.save_dir}/best_model.pth")
        
        # Save model info
        model_info = {
            'model_type': args.model_type,
            'parameters': count_parameters(model),
            'best_val_loss': history['best_val_loss'],
            'training_time': history['training_time'],
            'total_epochs': history['total_epochs'],
            'final_metrics': final_metrics
        }
        
        import json
        with open(os.path.join(args.save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📄 Model info saved to: {args.save_dir}/model_info.json")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())