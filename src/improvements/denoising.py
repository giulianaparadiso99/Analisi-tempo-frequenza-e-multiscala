"""
denoising.py - Tecniche di filtraggio per ridurre il rumore nei segnali DTMF
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.DTMF_implementation import recSequence, dialNumber
from src.experiments.noise import (
    generate_random_sequence,
    add_noise_to_signal,
    calculate_snr
)

def moving_average_denoise(x, M):
    """
    Denoising con media mobile di ordine M.
    
    Parameters
    ----------
    x : ndarray
        Segnale da filtrare
    M : int
        Ordine del filtro (larghezza finestra)
    
    Returns
    -------
    y : ndarray
        Segnale filtrato, stessa lunghezza di x
    """
    kernel = np.ones(M) / M
    y = np.convolve(x, kernel, mode="same")
    return y

def bandpass_filter_dtmf(x, Fs, lowcut=650, highcut=1700, order=5):
    """
    Filtro passa-banda per frequenze DTMF.
    Lascia passare solo le frequenze tra lowcut e highcut,
    attenuando rumore fuori dalla banda DTMF.

    Parameters
    ----------
    x : ndarray
        Segnale da filtrare
    Fs : int
        Frequenza di campionamento
    lowcut : float
        Frequenza di taglio bassa (default: 650 Hz)
    highcut : float
        Frequenza di taglio alta (default: 1700 Hz)
    order : int
        Ordine del filtro (default: 5)
    
    Returns
    -------
    y : ndarray
        Segnale filtrato
    """
    
    nyquist = 0.5 * Fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, x)
    
    return y

def leaky_integrator_denoise(x, lam):
    """
    Denoising con leaky integrator di primo ordine:
    Implementa la ricorrenza:
        y[n] = lam * y[n-1] + (1 - lam) * x[n]
    
    Parameters
    ----------
    x : ndarray
        Segnale da filtrare
    lam : float
        Parametro di memoria (0 < lam < 1)
        Valori vicini a 1: più smoothing
        Valori vicini a 0: meno smoothing
    
    Returns
    -------
    y : ndarray
        Segnale filtrato
    """
    y = np.zeros_like(x)
    y[0] = (1 - lam) * x[0]
    for n in range(1, len(x)):
        y[n] = lam * y[n-1] + (1 - lam) * x[n]
    return y


def test_denoising_methods(sigmas, duration=0.15, Fs=8000, n_sequences=100, 
                          M=5, lam=0.8):
    """
    Testa e confronta tutti i metodi di denoising.
    
   Parameters
    ----------
    sigmas : list
        Livelli di rumore da testare
    duration : float
        Durata dei toni
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze per test
    M : int
        Ordine media mobile
    lam : float
        Parametro leaky integrator
    
    Returns
    -------
    results : dict
        Accuratezze per ogni metodo
    """
    methods = {
        'no_filter': lambda x: x,
        'moving_average': lambda x: moving_average_denoise(x, M),
        'leaky_integrator': lambda x: leaky_integrator_denoise(x, lam),
        'bandpass': lambda x: bandpass_filter_dtmf(x, Fs)
    }
    
    results = {name: [] for name in methods.keys()}
    
    for sigma in sigmas:
        print(f"\nTesting sigma = {sigma}")
        
        for method_name, filter_func in methods.items():
            total = 0
            correct = 0
            
            for _ in range(n_sequences):
                seq = generate_random_sequence(10)
                true_seq = list(seq)
                signal_clean = dialNumber(seq, duration, Fs)
                
                if sigma == 0:
                    signal_noisy = signal_clean.copy()
                else:
                    signal_noisy, _ = add_noise_to_signal(signal_clean, sigma)
                
                # Applica filtro
                signal_filtered = filter_func(signal_noisy)
                
                # Riconosci
                detected = recSequence(signal_filtered, duration, Fs)
                
                total += 1
                if detected == true_seq:
                    correct += 1
            
            accuracy = 100 * correct / total
            results[method_name].append(accuracy)
            print(f"  {method_name}: {accuracy:.1f}%")
    
    return results


def plot_denoising_comparison(sigmas, results, save_path=None):
    """
    Confronta visivamente tutti i metodi di denoising.

    Parameters
    ----------
    sigmas : list
        Livelli di rumore testati
    results : dict
        Accuratezze per ogni metodo
    save_path : str, optional
        Percorso salvataggio grafico
    """
    plt.figure(figsize=(10, 6))
    
    labels = {
        'no_filter': 'Senza filtraggio',
        'moving_average': 'Media mobile (M=5)',
        'leaky_integrator': 'Leaky integrator (λ=0.8)',
        'bandpass': 'Passa-banda DTMF'
    }
    
    markers = {
        'no_filter': 'o',
        'moving_average': 's',
        'leaky_integrator': '^',
        'bandpass': 'd'
    }
    
    for method_name, accuracies in results.items():
        plt.plot(sigmas, accuracies, 
                marker=markers[method_name],
                linewidth=2, markersize=8,
                label=labels[method_name])
    
    plt.xlabel('Deviazione standard del rumore σ', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Confronto tecniche di denoising', fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def run_denoising_experiments(sigmas=None, Fs=8000, n_sequences=100, 
                              save_folder="Plots_denoising"):
    """
    Esegue tutti gli esperimenti di denoising.

    Parameters
    ----------
    sigmas : list, optional
        Livelli di rumore (default: [0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5, 6])
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze per test
    save_folder : str
        Cartella output
    
    Returns
    -------
    results : dict
        Risultati completi esperimenti
    """
    if sigmas is None:
        sigmas = [0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5, 6]
    
    os.makedirs(save_folder, exist_ok=True)
    
    print("ESPERIMENTI: TECNICHE DI DENOISING")
    
    # Test tutti i metodi
    results = test_denoising_methods(sigmas, Fs=Fs, n_sequences=n_sequences)
    
    # Plot confronto
    plot_denoising_comparison(sigmas, results,
                             save_path=os.path.join(save_folder, "denoising_comparison.png"))
    
    # Tabella risultati
    print("\nTABELLA RISULTATI")
    print(f"{'σ':<8} {'No filter':<12} {'Media mobile':<15} {'Leaky':<12} {'Passa-banda':<12}")
    print("-" * 70)
    for i, sigma in enumerate(sigmas):
        print(f"{sigma:<8.2f} "
              f"{results['no_filter'][i]:<12.1f} "
              f"{results['moving_average'][i]:<15.1f} "
              f"{results['leaky_integrator'][i]:<12.1f} "
              f"{results['bandpass'][i]:<12.1f}")
    
    print(f"\nRisultati salvati in: {save_folder}/")
    
    return results
