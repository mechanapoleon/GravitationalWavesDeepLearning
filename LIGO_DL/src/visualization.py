import matplotlib.pyplot as plt
import numpy as np
import os

class GWVisualizer:
    def __init__(self, save_path="plots"):
        """
        Args:
            save_path: directory dove salvare i plot
        """
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.ioff()  # Disabilita la modalità interattiva
    
    def plot_training_history(self, losses, model_name="model", save=True):
        """Visualizza l'andamento del training"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.title(f'Training History - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_path, f'training_history_{model_name}.png'))
        plt.close()

    def plot_training_comparison(self, losses_dict, save=True):
        """Confronta le loss di diversi modelli"""
        plt.figure(figsize=(12, 6))
        for model_name, losses in losses_dict.items():
            plt.plot(losses, label=f'{model_name} Loss')
        plt.title('Training History Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_path, 'training_comparison.png'))
        plt.close()
    
    def plot_learning_rates(self, initial_lr=0.01, epochs=50, model_name="optimized", save=True):
        """Plot del learning rate schedule"""
        epochs_range = np.arange(epochs)
        lr = initial_lr / (1 + 0.01 * epochs_range)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, lr, 'r-', label='Learning Rate')
        plt.title(f'Learning Rate Schedule - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_path, f'learning_rate_{model_name}.png'))
        plt.close()

    def plot_model_comparison(self, baseline_preds, optimized_preds, y_true, save=True):
        """Confronta le predizioni dei due modelli"""
        plt.figure(figsize=(10, 6))
        x = np.arange(len(y_true))
        width = 0.35
        
        plt.bar(x - width/2, baseline_preds, width, label='Baseline')
        plt.bar(x + width/2, optimized_preds, width, label='Optimized')
        plt.plot(x, y_true, 'r*', label='True Labels', markersize=10)
        
        plt.xlabel('Sample')
        plt.ylabel('Prediction')
        plt.title('Model Predictions Comparison')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'predictions_comparison.png'))
        plt.close()
        
    def plot_signal_comparison(self, signals, labels, predictions, model_name="model", save=True):
        """Visualizza e confronta i segnali"""
        n_signals = len(signals)
        fig, axs = plt.subplots(n_signals, 1, figsize=(15, 4*n_signals))
        
        if n_signals == 1:
            axs = [axs]
        
        for i, (signal, label, pred) in enumerate(zip(signals, labels, predictions)):
            axs[i].plot(signal, 'b-', alpha=0.6, label='Signal')
            axs[i].set_title(f'Signal {i+1} (True: {label}, Pred: {pred:.4f})')
            axs[i].set_xlabel('Time Sample')
            axs[i].set_ylabel('Amplitude')
            axs[i].grid(True)
            axs[i].legend()
        
        plt.suptitle(f'Signal Comparison - {model_name}')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_path, f'signal_comparison_{model_name}.png'))
        plt.close()
    
    def plot_confusion_matrix(self, true_labels, predictions, model_name="model", threshold=0.5, save=True):
        """Visualizza la matrice di confusione"""
        pred_labels = (predictions > threshold).astype(int)
        
        cm = np.zeros((2, 2))
        for t, p in zip(true_labels, pred_labels):
            cm[t][p] += 1
            
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, int(cm[i, j]),
                             ha="center", va="center", color="w" if cm[i, j] > 1 else "black")
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No GW', 'GW'])
        ax.set_yticklabels(['No GW', 'GW'])
        plt.colorbar(im, label='Number of Samples')
        if save:
            plt.savefig(os.path.join(self.save_path, f'confusion_matrix_{model_name}.png'))
        plt.close()
    
    def plot_signal_spectrum(self, signals, labels, predictions, model_name="model", save=True):
        """Visualizza lo spettro dei segnali"""
        n_signals = len(signals)
        fig, axs = plt.subplots(n_signals, 1, figsize=(15, 4*n_signals))
        
        if n_signals == 1:
            axs = [axs]
        
        for i, (signal, label, pred) in enumerate(zip(signals, labels, predictions)):
            spectrum = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal))
            
            pos_mask = freqs > 0
            axs[i].plot(freqs[pos_mask], spectrum[pos_mask], 'r-', alpha=0.6)
            axs[i].set_title(f'Spectrum {i+1} (True: {label}, Pred: {pred:.4f})')
            axs[i].set_xlabel('Normalized Frequency')
            axs[i].set_ylabel('Magnitude')
            axs[i].grid(True)
            axs[i].set_yscale('log')
        
        plt.suptitle(f'Signal Spectrum - {model_name}')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_path, f'signal_spectrum_{model_name}.png'))
        plt.close()