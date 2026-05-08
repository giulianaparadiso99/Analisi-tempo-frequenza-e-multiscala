import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.fft import fft, fftshift
from src.DTMF_implementation import recSequence, generate_random_sequence, dialNumber

def add_noise_to_signal(signal, sigma):
    """
    Aggiunge rumore gaussiano a un segnale.
    
    Parameters
    ----------
    signal : ndarray
        Segnale pulito
    sigma : float
        Deviazione standard del rumore gaussiano
    
    Returns
    -------
    signal_noisy : ndarray
        Segnale con rumore
    noise : ndarray
        Rumore aggiunto
    """
    noise = np.random.normal(0, sigma, len(signal))
    signal_noisy = signal + noise
    return signal_noisy, noise


def calculate_snr(signal_clean, signal_noisy=None, noise=None):
    """
    Calcola il rapporto segnale-rumore (SNR) in dB.
    
    Parameters
    ----------
    signal_clean : ndarray
        Segnale pulito
    signal_noisy : ndarray, optional
        Segnale con rumore (alternativo a noise)
    noise : ndarray, optional
        Rumore (alternativo a signal_noisy)
    
    Returns
    -------
    snr_db : float
        SNR in decibel
    """
    if noise is None:
        if signal_noisy is None:
            raise ValueError("Fornire signal_noisy o noise")
        noise = signal_noisy - signal_clean
    
    P_signal = np.mean(signal_clean ** 2)
    P_noise = np.mean(noise ** 2)
    
    if P_noise == 0:
        return np.inf
    
    snr_db = 10 * np.log10(P_signal / P_noise)
    return snr_db


def plot_signal_with_noise_comparison(signal_clean, sigmas, duration, Fs, 
                                      save_folder="Plots_noise"):
    """
    Genera grafici di confronto tempo/frequenza per diversi livelli di rumore.
    
    Parameters
    ----------
    signal_clean : ndarray
        Segnale DTMF pulito
    sigmas : list
        Lista di deviazioni standard da testare
    duration : float
        Durata del segnale
    Fs : int
        Frequenza di campionamento
    save_folder : str
        Cartella per salvare i plot
    """
    
    os.makedirs(save_folder, exist_ok=True)
    
    n_rows = len(sigmas)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    t = np.linspace(0, duration, len(signal_clean), endpoint=False)
    
    for idx, sigma in enumerate(sigmas):
        # Aggiungi rumore
        signal_noisy, noise = add_noise_to_signal(signal_clean, sigma)
        snr = calculate_snr(signal_clean, noise=noise)
        
        # DOMINIO DEL TEMPO
        ax_time = axes[idx, 0]
        ax_time.plot(t, signal_clean, label='Pulito', alpha=0.7, linewidth=1.5)
        ax_time.plot(t, signal_noisy, label='Rumoroso', alpha=0.7, linewidth=1)
        ax_time.set_title(f'Rumoroso (σ = {sigma}) - SNR = {snr:.1f} dB')
        ax_time.set_xlabel('Tempo [s]')
        ax_time.set_ylabel('Ampiezza')
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        
        # DOMINIO DELLA FREQUENZA
        ax_freq = axes[idx, 1]
        
        # Spettro pulito
        X_clean = fftshift(fft(signal_clean))
        freqs = np.linspace(-Fs/2, Fs/2, len(X_clean), endpoint=False)
        ax_freq.plot(freqs, np.abs(X_clean), label='Pulito', linewidth=1.5)
        
        # Spettro rumoroso
        X_noisy = fftshift(fft(signal_noisy))
        ax_freq.plot(freqs, np.abs(X_noisy), label='Rumoroso', alpha=0.7, linewidth=1)
        
        ax_freq.set_title(f'Spettro (σ = {sigma})')
        ax_freq.set_xlabel('Frequenza [Hz]')
        ax_freq.set_ylabel('|X(f)|')
        ax_freq.grid(True, alpha=0.3)
        ax_freq.legend()
        ax_freq.set_xlim(0, Fs/2)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, "signal_noise_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato: {save_path}")
    plt.show()
    plt.close()


def plot_noise_zoom_temporal(signal_clean, sigmas, Fs, zoom_start=0, zoom_duration=0.04,
                             save_folder="Plots_noise"):
    """
    Crea zoom temporale per mostrare effetto del rumore sulla forma d'onda.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Indici per lo zoom
    idx_start = int(zoom_start * Fs)
    idx_end = int((zoom_start + zoom_duration) * Fs)
    
    t_zoom = np.linspace(zoom_start, zoom_start + zoom_duration, idx_end - idx_start)
    signal_zoom = signal_clean[idx_start:idx_end]
    
    fig, axes = plt.subplots(len(sigmas), 1, figsize=(12, 2.5*len(sigmas)))
    
    if len(sigmas) == 1:
        axes = [axes]
    
    for idx, sigma in enumerate(sigmas):
        signal_noisy, _ = add_noise_to_signal(signal_clean, sigma)
        signal_noisy_zoom = signal_noisy[idx_start:idx_end]
        
        axes[idx].plot(t_zoom, signal_zoom, label='Pulito', linewidth=2)
        axes[idx].plot(t_zoom, signal_noisy_zoom, label=f'Rumore (σ={sigma})', 
                      alpha=0.8, linewidth=1.5)
        axes[idx].set_title(f'Zoom temporale - Rumore σ = {sigma}')
        axes[idx].set_xlabel('Tempo [s]')
        axes[idx].set_ylabel('Ampiezza')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, "noise_temporal_zoom.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico zoom salvato: {save_path}")
    plt.show()
    plt.close()

def test_noise_accuracy(sigmas, duration=0.15, Fs=8000, n_sequences=1000, n_trials=5):
    """
    Testa accuratezza al variare del rumore.
    
    Parameters
    ----------
    sigmas : list
        Lista di deviazioni standard del rumore
    duration : float
        Durata di ciascun tono
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze casuali diverse
    n_trials : int
        Numero di realizzazioni rumorose per sequenza
    
    Returns
    -------
    results : dict
        Dizionario con:
        - 'sigmas': list
        - 'accuracies': list (%)
        - 'snrs': list (dB)
    """

    accuracies = []
    snrs = []
    
    print("TEST ACCURATEZZA AL VARIARE DEL RUMORE")
    
    for sigma in sigmas:
        total = 0
        correct = 0
        snr_list = []
        
        print(f"\nσ = {sigma}")
        
        for _ in range(n_sequences):
            # Genera sequenza casuale
            seq = generate_random_sequence(10)
            true_seq = list(seq)
            
            # Segnale pulito
            signal_clean = dialNumber(seq, duration, Fs)
            
            for _ in range(n_trials):
                # Aggiungi rumore
                if sigma == 0:
                    signal_noisy = signal_clean.copy()
                    snr = np.inf
                else:
                    signal_noisy, noise = add_noise_to_signal(signal_clean, sigma)
                    snr = calculate_snr(signal_clean, noise=noise)
                
                snr_list.append(snr)
                
                # Riconosci
                detected = recSequence(signal_noisy, duration, Fs)
                
                total += 1
                if detected == true_seq:
                    correct += 1
        
        accuracy = 100 * correct / total
        avg_snr = np.mean([s for s in snr_list if np.isfinite(s)])
        
        accuracies.append(accuracy)
        snrs.append(avg_snr if np.isfinite(avg_snr) else np.inf)
        
        print(f"Accuratezza: {accuracy:.1f}% ({correct}/{total})")
        print(f"SNR medio: {avg_snr:.1f} dB")
    
    return {
        'sigmas': sigmas,
        'accuracies': accuracies,
        'snrs': snrs
    }


def plot_accuracy_vs_sigma(sigmas, accuracies, save_path=None):
    """
    Grafico accuratezza vs σ.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, accuracies, 'o-', linewidth=2, markersize=8, color='#E63946')
    plt.xlabel('Deviazione standard del rumore σ', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Accuratezza di riconoscimento DTMF\nin funzione del rumore', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_accuracy_vs_snr(snrs, accuracies, save_path=None):
    """
    Grafico accuratezza vs SNR in dB.
    """
    # Filtra eventuali inf
    finite_mask = np.isfinite(snrs)
    snrs_finite = np.array(snrs)[finite_mask]
    acc_finite = np.array(accuracies)[finite_mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snrs_finite, acc_finite, 'o-', linewidth=2, markersize=8, color='#F77F00')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Accuratezza di riconoscimento DTMF\nin funzione del SNR', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    # Inverti asse x (SNR alto = poco rumore = sinistra)
    plt.gca().invert_xaxis()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def test_noise_vs_duration(durations, sigmas, Fs=8000, n_sequences=50):
    """
    Testa accuratezza per matrice durata × rumore.

    Parameters
    ----------
    durations : list
        Durate da testare
    sigmas : list
        Livelli di rumore da testare
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze per test
    
    Returns
    -------
    accuracy_matrix : ndarray
        Matrice (len(durations) × len(sigmas))
    """

    accuracy_matrix = np.zeros((len(durations), len(sigmas)))
    
    print("TEST ACCURATEZZA: DURATA × RUMORE")
    
    for i, duration in enumerate(durations):
        print(f"\n--- Durata: {duration}s ---")
        
        for j, sigma in enumerate(sigmas):
            total = 0
            correct = 0
            
            for _ in range(n_sequences):
                # Genera sequenza
                seq = generate_random_sequence(10)
                true_seq = list(seq)
                
                # Genera segnale
                signal_clean = dialNumber(seq, duration, Fs)
                
                # Aggiungi rumore
                if sigma == 0:
                    signal_noisy = signal_clean
                else:
                    signal_noisy, _ = add_noise_to_signal(signal_clean, sigma)
                
                # Riconosci
                detected = recSequence(signal_noisy, duration, Fs)
                
                total += 1
                if detected == true_seq:
                    correct += 1
            
            accuracy = 100 * correct / total
            accuracy_matrix[i, j] = accuracy
            print(f"  σ={sigma}: {accuracy:.1f}%")
    
    return accuracy_matrix


def plot_duration_noise_heatmap(durations, sigmas, accuracy_matrix, save_path=None):
    """
    Heatmap durata × rumore.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[f'{s}' for s in sigmas],
                yticklabels=[f'{d}s' for d in durations],
                cbar_kws={'label': 'Accuratezza [%]'},
                vmin=0, vmax=100)
    
    plt.title('Accuratezza riconoscimento DTMF\nvs Durata e Livello di Rumore', 
              fontsize=14, pad=20)
    plt.xlabel('Deviazione standard rumore σ', fontsize=12)
    plt.ylabel('Durata tono', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap salvata: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def run_noise_experiments(sigmas=None, durations=None, Fs=8000, 
                         n_sequences=100, save_folder="Plots_noise"):
    """
    Esegue tutti gli esperimenti sul rumore.
    Genera:
    1. Confronto segnale pulito/rumoroso (tempo + frequenza)
    2. Zoom temporale
    3. Grafico accuratezza vs σ
    4. Grafico accuratezza vs SNR
    5. Tabella accuratezza per ogni σ
    6. Heatmap durata × rumore

    Parameters
    ----------
    sigmas : list, optional
        Livelli di rumore (default: [0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5, 6])
    durations : list, optional
        Durate per heatmap (default: [0.05, 0.1, 0.15, 0.2, 0.5])
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze per test
    save_folder : str
        Cartella output
    
    Returns
    -------
    results : dict
        Tutti i risultati degli esperimenti
    """
    if sigmas is None:
        sigmas = [0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5, 6]
    
    if durations is None:
        durations = [0.05, 0.1, 0.15, 0.2, 0.5]
    
    os.makedirs(save_folder, exist_ok=True)
    
    print("ESPERIMENTI: EFFETTO DEL RUMORE")
    
    # PARTE 1: Visualizzazioni segnale
    print("\n1. Generazione grafici tempo/frequenza...")
    
    # Genera segnale esempio
    seq_example = "3456318060"
    signal_clean = dialNumber(seq_example, 0.15, Fs)
    
    # Confronto con diversi livelli di rumore
    sigmas_plot = [0.05, 0.2, 0.5, 1, 2]
    plot_signal_with_noise_comparison(signal_clean, sigmas_plot, 0.15, Fs, save_folder)
    
    # Zoom temporale
    plot_noise_zoom_temporal(signal_clean, sigmas_plot, Fs, 
                            zoom_start=0.1, zoom_duration=0.04, 
                            save_folder=save_folder)
    
    # PARTE 2: Test accuratezza
    
    print("\n2. Test accuratezza al variare del rumore...")
    
    results_acc = test_noise_accuracy(sigmas, duration=0.15, Fs=Fs, 
                                     n_sequences=n_sequences, n_trials=5)
    
    # Grafico accuratezza vs σ
    plot_accuracy_vs_sigma(results_acc['sigmas'], results_acc['accuracies'],
                          save_path=os.path.join(save_folder, "accuracy_vs_sigma.png"))
    
    # Grafico accuratezza vs SNR
    plot_accuracy_vs_snr(results_acc['snrs'], results_acc['accuracies'],
                        save_path=os.path.join(save_folder, "accuracy_vs_snr.png"))
    
    # Tabella risultati
    print("TABELLA RISULTATI")
    print(f"{'σ':<8} {'SNR [dB]':<12} {'Accuratezza [%]':<15}")
    print("-" * 40)
    for sigma, snr, acc in zip(results_acc['sigmas'], results_acc['snrs'], 
                               results_acc['accuracies']):
        snr_str = f"{snr:.1f}" if np.isfinite(snr) else "∞"
        print(f"{sigma:<8.2f} {snr_str:<12} {acc:<15.1f}")
    
    # PARTE 3: Matrice durata × rumore
    
    print("\n3. Test matrice durata × rumore...")
    
    sigmas_heatmap = [0, 0.5, 1, 2, 3, 4]
    accuracy_matrix = test_noise_vs_duration(durations, sigmas_heatmap, Fs, n_sequences=50)
    
    plot_duration_noise_heatmap(durations, sigmas_heatmap, accuracy_matrix,
                               save_path=os.path.join(save_folder, "duration_noise_heatmap.png"))
    
    print("\n" + "="*60)
    print(f"Tutti i risultati salvati in: {save_folder}/")
    print("="*60 + "\n")
    
    return {
        'accuracy_results': results_acc,
        'accuracy_matrix': accuracy_matrix,
        'durations': durations,
        'sigmas_heatmap': sigmas_heatmap
    }