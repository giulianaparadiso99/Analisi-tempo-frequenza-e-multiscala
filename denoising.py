"""
denoising.py - Tecniche di filtraggio per ridurre il rumore nei segnali DTMF
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.DTMF_implementation import tone, recSequence, dialNumber
from src.experiments.noise import add_noise_to_signal, calculate_snr
from src.experiments.noise import generate_random_sequence

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

def compare_filter_denoising(filter_name, filter_func, tone_char, duration, 
                             sigmas, Fs=8000, save_path=None):
    """
    Confronta effetto di UN filtro per diversi livelli di rumore.
    
    Parameters
    ----------
    filter_name : str
        Nome del filtro (per titolo)
    filter_func : callable
        Funzione filtro da applicare (es. moving_average_denoise)
    tone_char : str
        Tono DTMF da testare (es. '5')
    duration : float
        Durata del tono in secondi
    sigmas : list
        Livelli di rumore da visualizzare (es. [0.2, 0.5, 1.0])
    Fs : int
        Frequenza di campionamento
    save_path : str, optional
        Percorso per salvare il plot
    """
    n_sigmas = len(sigmas)
    fig, axes = plt.subplots(n_sigmas, 2, figsize=(14, 4*n_sigmas))
    
    # Genera segnale pulito (una volta sola)
    t, signal_clean = tone(tone_char, duration, Fs)
    N = len(signal_clean)
    
    for i, sigma in enumerate(sigmas):
        # Aggiungi rumore
        signal_noisy, _ = add_noise_to_signal(signal_clean, sigma)
        
        # Applica filtro
        signal_filtered = filter_func(signal_noisy)
        
        # SUBPLOT TEMPO
        ax_time = axes[i, 0] if n_sigmas > 1 else axes[0]
        
        ax_time.plot(t, signal_clean, 'g-', linewidth=1.5, alpha=0.8, label='Originale')
        ax_time.plot(t, signal_noisy, 'r-', linewidth=0.5, alpha=0.6, label=f'Rumoroso (σ={sigma})')
        ax_time.plot(t, signal_filtered, 'b-', linewidth=1.2, alpha=0.8, label='Filtrato')
        
        ax_time.set_xlabel('Tempo [s]')
        ax_time.set_ylabel('Ampiezza')
        ax_time.set_title(f'Dominio del tempo - σ={sigma}')
        ax_time.legend(loc='upper right', fontsize=9)
        ax_time.grid(True, alpha=0.3)
        
        # SUBPLOT SPETTRO
        ax_freq = axes[i, 1] if n_sigmas > 1 else axes[1]
        
        # Calcola FFT
        X_clean = np.fft.fft(signal_clean)
        X_noisy = np.fft.fft(signal_noisy)
        X_filtered = np.fft.fft(signal_filtered)
        
        freqs = np.fft.fftfreq(N, 1/Fs)
        
        # Prendi solo metà positiva
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        
        ax_freq.plot(freqs_pos, np.abs(X_clean[pos_mask]), 'g-', linewidth=1.5, alpha=0.8, label='Originale')
        ax_freq.plot(freqs_pos, np.abs(X_noisy[pos_mask]), 'r-', linewidth=0.5, alpha=0.6, label='Rumoroso')
        ax_freq.plot(freqs_pos, np.abs(X_filtered[pos_mask]), 'b-', linewidth=1.2, alpha=0.8, label='Filtrato')
        
        ax_freq.set_xlabel('Frequenza [Hz]')
        ax_freq.set_ylabel('|X(f)|')
        ax_freq.set_title(f'Dominio della frequenza - σ={sigma}')
        ax_freq.set_xlim(0, 2500)
        ax_freq.legend(loc='upper right', fontsize=9)
        ax_freq.grid(True, alpha=0.3)
    
    plt.suptitle(f'Effetto del filtro: {filter_name}', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Plot salvato: {save_path}")
    
    plt.show()
    plt.close()


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
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def run_denoising_experiments(sigmas=None, duration=0.15, Fs=8000, 
                              tone_char='9',n_sequences=100,
                              sigmas_visual=[0.2, 0.5, 1.0],
                              plot_folder="Plots/improvements/denoising"):
    if sigmas is None:
        sigmas = [0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5, 6]
    
    os.makedirs(plot_folder, exist_ok=True)
    print("ESPERIMENTI: TECNICHE DI DENOISING")
    
    # Confronti visivi
    print("\n1. Generazione confronti visuali per filtri...")
    
    # Plot per media mobile
    compare_filter_denoising(
        filter_name="Media mobile (M=5)",
        filter_func=lambda x: moving_average_denoise(x, M=5),
        tone_char=tone_char,
        duration=duration,
        sigmas=sigmas_visual,
        Fs=Fs,
        save_path=os.path.join(plot_folder, "denoising_moving_average.pdf")
    )
    
    # Plot per leaky integrator
    compare_filter_denoising(
        filter_name="Leaky integrator (λ=0.8)",
        filter_func=lambda x: leaky_integrator_denoise(x, lam=0.8),
        tone_char=tone_char,
        duration=duration,
        sigmas=sigmas_visual,
        Fs=Fs,
        save_path=os.path.join(plot_folder, "denoising_leaky_integrator.pdf")
    )
    
    # Plot per passa-banda
    compare_filter_denoising(
        filter_name="Passa-banda DTMF (650-1700 Hz)",
        filter_func=lambda x: bandpass_filter_dtmf(x, Fs),
        tone_char=tone_char,
        duration=duration,
        sigmas=sigmas_visual,
        Fs=Fs,
        save_path=os.path.join(plot_folder, "denoising_bandpass.pdf")
    )
    
    print("ESPERIMENTI: TECNICHE DI DENOISING")
    
    # Test tutti i metodi
    results = test_denoising_methods(sigmas, Fs=Fs, n_sequences=n_sequences)
    
    # Plot confronto
    plot_denoising_comparison(sigmas, results,
                             save_path=os.path.join(plot_folder, "denoising_comparison.pdf"))
    
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
    
    print(f"\nRisultati salvati in: {plot_folder}/")
    
    return results
