"""
zero_padding.py - Analisi dell'effetto dello zero-padding sul riconoscimento DTMF
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.fft import fft, fftshift
from DTMF_implementation import (
    tone,
    dialNumber,
    F1_DEFAULT,
    F2_DEFAULT,
    TONES_DEFAULT
)
from src.experiments.noise import generate_random_sequence

def detect_single_tone_with_zeropadding(x, Fs, M=None, F1=None, F2=None, tones=None):
    """
    Riconosce un singolo tono DTMF usando FFT con zero-padding.

    Parameters
    ----------
    x : ndarray
        Segnale del tono
    Fs : int
        Frequenza di campionamento
    M : int, optional
        Lunghezza della FFT (con zero-padding se M > len(x))
        Se None, usa len(x) (nessun zero-padding)
    F1 : ndarray, optional
        Frequenze basse (default: ETSI standard)
    F2 : ndarray, optional
        Frequenze alte (default: ETSI standard)
    tones : dict, optional
        Dizionario toni (default: ETSI standard)
        
    Returns
    -------
    key : str or None
        Tasto riconosciuto
    """
    if F1 is None:
        F1 = F1_DEFAULT
    if F2 is None:
        F2 = F2_DEFAULT
    if tones is None:
        tones = TONES_DEFAULT
    
    N = len(x)
    
    # Se M non specificato o M <= N, usa N (no zero-padding)
    if M is None or M <= N:
        M = N
    
    # FFT con zero-padding
    X = fft(x, n=M)
    X_shifted = fftshift(X)
    freqs = np.linspace(-Fs/2, Fs/2, M, endpoint=False)
    
    mag = np.abs(X_shifted)
    
    # Solo frequenze positive
    mask_pos = freqs >= 0
    freqs_pos = freqs[mask_pos]
    mag_pos = mag[mask_pos]
    
    # Trova frequenza bassa
    amps_F1 = []
    for f in F1:
        idx = np.argmin(np.abs(freqs_pos - f))
        amps_F1.append(mag_pos[idx])
    i = int(np.argmax(amps_F1))
    
    # Trova frequenza alta
    amps_F2 = []
    for f in F2:
        idx = np.argmin(np.abs(freqs_pos - f))
        amps_F2.append(mag_pos[idx])
    j = int(np.argmax(amps_F2))
    
    # Mappa al tasto
    for key, (r, c) in tones.items():
        if r == i and c == j:
            return key
    
    return None


def recSequence_with_zeropadding(signal, toneDuration, Fs, M=None):
    """
    Riconosce una sequenza di toni con zero-padding.

    Parameters
    ----------
    signal : ndarray
        Segnale della sequenza
    toneDuration : float
        Durata di ciascun tono
    Fs : int
        Frequenza di campionamento
    M : int, optional
        Lunghezza FFT per zero-padding
        
    Returns
    -------
    detected : list
        Lista dei tasti riconosciuti
    """
    samples_per_tone = int(Fs * toneDuration)
    N = len(signal)
    
    detected = []
    
    for start in range(0, N, samples_per_tone):
        end = start + samples_per_tone
        x = signal[start:end]
        
        if len(x) < samples_per_tone:
            break
        
        key = detect_single_tone_with_zeropadding(x, Fs, M=M)
        detected.append(key)
    
    return detected


def compare_spectrum_with_zeropadding(tone_char, duration, Fs, M_factors=[1, 2, 4, 8],
                                     save_folder="Plots_zeropadding"):
    """
    Confronta lo spettro di un tono con diversi livelli di zero-padding.

    Parameters
    ----------
    tone_char : str
        Tasto da analizzare
    duration : float
        Durata del tono
    Fs : int
        Frequenza di campionamento
    M_factors : list
        Fattori di zero-padding (M = factor * len(x))
    save_folder : str
        Cartella output
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Genera il tono
    t, x = tone(tone_char, duration, Fs)
    N = len(x)
    
    n_plots = len(M_factors)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3*n_plots))
    
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, factor in enumerate(M_factors):
        M = factor * N
        
        # Tempo
        ax_time = axes[idx, 0]
        if factor == 1:
            ax_time.plot(t, x, linewidth=1)
        else:
            t_padded = np.arange(M) / Fs
            x_padded = np.zeros(M)
            x_padded[:N] = x
            ax_time.plot(t_padded, x_padded, linewidth=1)
        
        ax_time.set_title(f"Tono '{tone_char}' - Zero-padding {factor}x (M={M})")
        ax_time.set_xlabel('Tempo [s]')
        ax_time.set_ylabel('Ampiezza')
        ax_time.grid(True, alpha=0.3)
        
        # Frequenza
        ax_freq = axes[idx, 1]
        X = fft(x, n=M)
        X_shifted = fftshift(X)
        freqs = np.linspace(-Fs/2, Fs/2, M, endpoint=False)
        
        ax_freq.plot(freqs, np.abs(X_shifted), linewidth=1)
        ax_freq.set_title(f"Spettro - Zero-padding {factor}x")
        ax_freq.set_xlabel('Frequenza [Hz]')
        ax_freq.set_ylabel('|X(f)|')
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(0, Fs/2)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"zeropadding_comparison_{tone_char}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato: {save_path}")
    plt.show()
    plt.close()


def test_zeropadding_vs_duration(durations, M_factors, Fs=8000, n_sequences=50):
    """
    Testa l'accuratezza con zero-padding al variare della durata e di M.

    Parameters
    ----------
    durations : list
        Durate da testare
    M_factors : list
        Fattori di zero-padding
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze per test
        
    Returns
    -------
    accuracy_matrix : ndarray
        Matrice (len(durations) × len(M_factors))
    """

    accuracy_matrix = np.zeros((len(durations), len(M_factors)))
    
    print("\nTEST ACCURATEZZA: DURATA vs ZERO-PADDING")
    
    for i, duration in enumerate(durations):
        print(f"\nDurata: {duration}s")
        samples_per_tone = int(Fs * duration)
        
        for j, factor in enumerate(M_factors):
            M = factor * samples_per_tone
            total = 0
            correct = 0
            
            for _ in range(n_sequences):
                seq = generate_random_sequence(10)
                true_seq = list(seq)
                
                signal = dialNumber(seq, duration, Fs)
                detected = recSequence_with_zeropadding(signal, duration, Fs, M=M)
                
                total += 1
                if detected == true_seq:
                    correct += 1
            
            accuracy = 100 * correct / total
            accuracy_matrix[i, j] = accuracy
            print(f"  M = {factor}x: {accuracy:.1f}%")
    
    return accuracy_matrix


def plot_zeropadding_heatmap(durations, M_factors, accuracy_matrix, save_path=None):
    """
    Heatmap accuratezza vs durata e zero-padding.

    Parameters
    ----------
    durations : list
        Durate testate
    M_factors : list
        Fattori zero-padding testati
    accuracy_matrix : ndarray
        Matrice accuratezze
    save_path : str, optional
        Percorso salvataggio
    """
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[f'{m}x' for m in M_factors],
                yticklabels=[f'{d}s' for d in durations],
                cbar_kws={'label': 'Accuratezza [%]'},
                vmin=0, vmax=100)
    
    plt.title('Accuratezza riconoscimento DTMF\nvs Durata e Zero-padding', 
              fontsize=14, pad=20)
    plt.xlabel('Fattore zero-padding (M)', fontsize=12)
    plt.ylabel('Durata tono', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap salvata: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def run_zeropadding_experiments(durations=None, M_factors=None, Fs=8000,
                                n_sequences=100, save_folder="Plots_zeropadding"):
    """
    Esegue tutti gli esperimenti sullo zero-padding.
    Genera:
    1. Confronti visivi spettro con/senza zero-padding
    2. Test accuratezza vs durata e M
    3. Heatmap durata × M

    Parameters
    ----------
    durations : list, optional
        Durate da testare (default: [0.02, 0.05, 0.1, 0.15, 0.2])
    M_factors : list, optional
        Fattori zero-padding (default: [1, 2, 4, 8])
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero sequenze per test
    save_folder : str
        Cartella output
    
    Returns
    -------
    results : dict
        Risultati completi
    """
    if durations is None:
        durations = [0.02, 0.05, 0.1, 0.15, 0.2]
    
    if M_factors is None:
        M_factors = [1, 2, 4, 8]
    
    os.makedirs(save_folder, exist_ok=True)
    
    print("ESPERIMENTI: ZERO-PADDING")
    
    # Parte 1: Visualizzazione spettri
    print("\n1. Generazione confronti spettrali...")
    
    test_tones = ['3', 'A', '*']
    for tone_char in test_tones:
        compare_spectrum_with_zeropadding(tone_char, 0.15, Fs, M_factors, save_folder)
    
    # Parte 2: Test accuratezza
    print("\n2. Test accuratezza vs durata e zero-padding...")
    
    accuracy_matrix = test_zeropadding_vs_duration(durations, M_factors, Fs, n_sequences)
    
    # Plot heatmap
    plot_zeropadding_heatmap(durations, M_factors, accuracy_matrix,
                            save_path=os.path.join(save_folder, "zeropadding_heatmap.png"))
    
    # Tabella risultati
    print("\nTABELLA RISULTATI")
    print(f"{'Durata':<10}", end='')
    for factor in M_factors:
        print(f"{'M=' + str(factor) + 'x':<12}", end='')
    print()
    print("-" * (10 + 12 * len(M_factors)))
    
    for i, duration in enumerate(durations):
        print(f"{duration:<10.2f}", end='')
        for j in range(len(M_factors)):
            print(f"{accuracy_matrix[i, j]:<12.1f}", end='')
        print()
    
    print(f"\nRisultati salvati in: {save_folder}/")
    
    return {
        'accuracy_matrix': accuracy_matrix,
        'durations': durations,
        'M_factors': M_factors
    }