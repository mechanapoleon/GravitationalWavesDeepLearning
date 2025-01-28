import json
import numpy as np
from datetime import datetime
import os

class DataLogger:
    def __init__(self, output_dir="results"):
        """
        Args:
            output_dir: directory dove salvare i risultati
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crea la directory se non esiste
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def save_training_results(self, model_params, losses, predictions, true_labels, signals):
        """
        Salva tutti i risultati del training per entrambi i modelli
        
        Args:
            model_params: dizionario con i parametri di entrambi i modelli
            losses: dizionario con le losses di entrambi i modelli
            predictions: dizionario con le predizioni di entrambi i modelli
            true_labels: array con le etichette vere
            signals: array con i segnali
        """
        results = {
            'timestamp': self.timestamp,
            'model_parameters': model_params,  # Già strutturato come {'baseline': {...}, 'optimized': {...}}
            'training': {
                'losses': losses,  # Già strutturato come {'baseline': [...], 'optimized': [...]}
                'predictions': {
                    'baseline': predictions['baseline'].tolist() if isinstance(predictions['baseline'], np.ndarray) else predictions['baseline'],
                    'optimized': predictions['optimized'].tolist() if isinstance(predictions['optimized'], np.ndarray) else predictions['optimized']
                },
                'true_labels': true_labels.tolist() if isinstance(true_labels, np.ndarray) else true_labels
            },
            'performance': {
                'baseline': {
                    'final_loss': losses['baseline'][-1] if isinstance(losses['baseline'][-1], (int, float)) else float(losses['baseline'][-1]),
                    'accuracy': float(np.mean((np.array(predictions['baseline']) > 0.5) == np.array(true_labels)))
                },
                'optimized': {
                    'final_loss': losses['optimized'][-1] if isinstance(losses['optimized'][-1], (int, float)) else float(losses['optimized'][-1]),
                    'accuracy': float(np.mean((np.array(predictions['optimized']) > 0.5) == np.array(true_labels)))
                }
            }
        }
        
        # Salva i risultati in JSON
        results_file = os.path.join(self.output_dir, f'results_{self.timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Salva i segnali in formato numpy
        signals_file = os.path.join(self.output_dir, f'signals_{self.timestamp}.npy')
        np.save(signals_file, signals)
        
        print(f"\nRisultati salvati in:\n{results_file}\n{signals_file}")
        
        # Stampa anche un sommario delle performance
        print("\nSommario Performance:")
        print("Baseline Model:")
        print(f"  - Final Loss: {results['performance']['baseline']['final_loss']:.4f}")
        print(f"  - Accuracy: {results['performance']['baseline']['accuracy']*100:.2f}%")
        print("Optimized Model:")
        print(f"  - Final Loss: {results['performance']['optimized']['final_loss']:.4f}")
        print(f"  - Accuracy: {results['performance']['optimized']['accuracy']*100:.2f}%")