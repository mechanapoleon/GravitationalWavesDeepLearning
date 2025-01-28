import numpy as np
from gwpy.timeseries import TimeSeries

class LIGODataPreprocessor:
    """Classe per il preprocessing dei dati LIGO."""
    
    def __init__(self, window_size=32):
        """
        Inizializza il preprocessore.
        
        Args:
            window_size (int): Dimensione della finestra in secondi per ogni campione
        """
        self.window_size = window_size
    
    def load_strain_data(self, t0, detector='H1'):
        """
        Carica i dati grezzi da LIGO.
        
        Args:
            t0 (float): Tempo GPS centrale
            detector (str): Detector da usare ('H1', 'L1', o 'V1')
            
        Returns:
            TimeSeries: Dati grezzi dal detector
        """
        center = int(t0)
        half_window = self.window_size // 2
        return TimeSeries.fetch_open_data(detector, 
                                        center - half_window, 
                                        center + half_window)
    
    def preprocess_strain(self, strain):
        """
        Applica il preprocessing ai dati (whitening e bandpass).
        
        Args:
            strain (TimeSeries): Dati grezzi
            
        Returns:
            np.array: Dati preprocessati
        """
        # Whitening
        white_data = strain.whiten()
        # Bandpass filter tra 30 Hz e 400 Hz
        bp_data = white_data.bandpass(30, 400)
        return bp_data.value
    
    def create_sample(self, t0, detector='H1'):
        """
        Crea un singolo campione preprocessato.
        
        Args:
            t0 (float): Tempo GPS dell'evento
            detector (str): Detector da usare
            
        Returns:
            np.array: Dati preprocessati
        """
        strain = self.load_strain_data(t0, detector)
        return self.preprocess_strain(strain)
    
    def create_dataset(self, event_times, non_event_times, detector='H1'):
        """
        Crea un dataset di eventi e non-eventi.
        
        Args:
            event_times (list): Lista di tempi GPS degli eventi
            non_event_times (list): Lista di tempi GPS dei non-eventi
            detector (str): Detector da usare
            
        Returns:
            tuple: (X, y) dove X sono i dati e y le etichette
        """
        X = []
        y = []
        
        # Processa gli eventi (etichetta 1)
        for t in event_times:
            try:
                sample = self.create_sample(t, detector)
                X.append(sample)
                y.append(1)
            except Exception as e:
                print(f"Errore nel processare l'evento al tempo {t}: {e}")
        
        # Processa i non-eventi (etichetta 0)
        for t in non_event_times:
            try:
                sample = self.create_sample(t, detector)
                X.append(sample)
                y.append(0)
            except Exception as e:
                print(f"Errore nel processare il non-evento al tempo {t}: {e}")
        
        return np.array(X), np.array(y)

# Test del preprocessore
if __name__ == "__main__":
    # Tempo di GW150914
    gw150914_time = 1126259462.4
    
    # Creiamo un preprocessore con finestra di 32 secondi
    preprocessor = LIGODataPreprocessor(window_size=32)
    
    # Test con un singolo evento
    try:
        sample = preprocessor.create_sample(gw150914_time)
        print("Forma del campione:", sample.shape)
        print("Primi 10 valori:", sample[:10])
    except Exception as e:
        print(f"Errore durante il test: {e}")