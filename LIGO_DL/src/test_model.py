import numpy as np
from preprocessing import LIGODataPreprocessor
from model import DeepGWDetector, OptimizedGWDetector
from visualization import GWVisualizer
from data_logger import DataLogger

def main():
    # Inizializza il preprocessore e il visualizer
    preprocessor = LIGODataPreprocessor(window_size=32)
    visualizer = GWVisualizer(save_path="plots")
    
    # Inizializza il logger
    logger = DataLogger()

    # Tempi di esempio
    event_time = 1126259462.4  # GW150914
    non_event_times = [
        1126259462.4 + 100,
        1126259462.4 - 100
    ]
    
    print("Creazione del dataset...")
    X, y = preprocessor.create_dataset([event_time], non_event_times)
    
    # Normalizziamo i dati
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    
    print("Forma del dataset:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Parametri comuni
    input_dim = X.shape[1]
    hidden_dims = [64, 32]
    
    # Inizializziamo i modelli
    print("\nInizio training baseline model...")
    baseline_model = DeepGWDetector(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        learning_rate=0.001,  # Learning rate più basso per stabilità
        l2_reg=0.01
    )
    
    print("\nInizio training optimized model...")
    optimized_model = OptimizedGWDetector(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        initial_learning_rate=0.005,  # Learning rate iniziale più alto
        momentum=0.95,
        l2_reg=0.005
    )
    
    # Training dei modelli
    baseline_losses = baseline_model.train(X, y, epochs=50)
    baseline_preds = baseline_model.forward(X)
    print("\nPredizioni baseline model:")
    for i, pred in enumerate(baseline_preds):
        print(f"Sample {i}: True label = {y[i]}, Predicted probability = {pred[0]:.4f}")
    
    optimized_losses = optimized_model.train(X, y, epochs=50)
    optimized_preds = optimized_model.forward(X)
    print("\nPredizioni optimized model:")
    for i, pred in enumerate(optimized_preds):
        print(f"Sample {i}: True label = {y[i]}, Predicted probability = {pred[0]:.4f}")
    
    # Visualizzazioni
    # Plot delle loss
    visualizer.plot_training_history(baseline_losses, "baseline")
    visualizer.plot_training_history(optimized_losses, "optimized")
    
    # Plot confronto training
    visualizer.plot_training_comparison({
        'Baseline': baseline_losses,
        'Optimized': optimized_losses
    })
    
    # Plot dei segnali per ogni modello
    visualizer.plot_signal_comparison(X, y, baseline_preds.flatten(), "baseline")
    visualizer.plot_signal_comparison(X, y, optimized_preds.flatten(), "optimized")
    
    # Plot delle matrici di confusione
    visualizer.plot_confusion_matrix(y, baseline_preds.flatten(), "baseline")
    visualizer.plot_confusion_matrix(y, optimized_preds.flatten(), "optimized")
    
    # Plot degli spettri
    visualizer.plot_signal_spectrum(X, y, baseline_preds.flatten(), "baseline")
    visualizer.plot_signal_spectrum(X, y, optimized_preds.flatten(), "optimized")
    
    # Plot confronto predizioni
    visualizer.plot_model_comparison(
        baseline_preds.flatten(),
        optimized_preds.flatten(),
        y
    )
    
    # Salvataggio risultati
    logger.save_training_results(
        model_params={
            'baseline': {
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'learning_rate': 0.001,
                'l2_reg': 0.01
            },
            'optimized': {
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'initial_learning_rate': 0.005,
                'momentum': 0.95,
                'l2_reg': 0.005
            }
        },
        losses={
            'baseline': baseline_losses,
            'optimized': optimized_losses
        },
        predictions={
            'baseline': baseline_preds.flatten(),
            'optimized': optimized_preds.flatten()
        },
        true_labels=y,
        signals=X
    )

if __name__ == "__main__":
    main()