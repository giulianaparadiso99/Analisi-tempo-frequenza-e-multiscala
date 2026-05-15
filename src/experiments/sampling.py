"""
Analisi dell'effetto della frequenza di campionamento sui segnali DTMF.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample, butter, filtfilt
from scipy.fft import fft, fftshift
from src.DTMF_implementation import tone, dialNumber, recSequence
from src.experiments.duration import generate_random_sequence


def resample_signal(signal, Fs_original, Fs_target):
    """
    Ricampiona un segnale a una nuova frequenza di campionamento.
    
    Parameters
    ----------
    signal : ndarray
        Segnale originale
    Fs_original : int
        Frequenza di campionamento originale
    Fs_target : int
        Frequenza di campionamento target
        
    Returns
    -------
    signal_resampled : ndarray
        Segnale ricampionato
    """
    num_samples = int(len(signal) * Fs_target / Fs_original)
    signal_resampled = resample(signal, num_samples)
    return signal_resampled


def lowpass_antialias_filter(signal, Fs, cutoff_freq, order=5):
    """
    Applica un filtro passa-basso anti-aliasing.
    
    Parameters
    ----------
    signal : ndarray
        Segnale da filtrare
    Fs : int
        Frequenza di campionamento
    cutoff_freq : float
        Frequenza di taglio
    order : int
        Ordine del filtro
        
    Returns
    -------
    signal_filtered : ndarray
        Segnale filtrato
    """
    nyquist = 0.5 * Fs
    normalized_cutoff = cutoff_freq / nyquist
    
    b, a = butter(order, normalized_cutoff, btype='low')
    signal_filtered = filtfilt(b, a, signal)
    
    return signal_filtered


def compare_sampling_frequencies_visual(tone_char, duration, Fs_list, 
                                       save_path=None):
    """
    Confronta visivamente un tono a diverse frequenze di campionamento.
    
    Parameters
    ----------
    tone_char : str
        Tasto da analizzare
    duration : float
        Durata del tono
    Fs_list : list
        Lista di frequenze di campionamento da testare
    save_path : str
        Percorso per salvare il grafico
    """
    
    n_plots = len(Fs_list)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3*n_plots))
    
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, Fs in enumerate(Fs_list):
        # Genera il tono
        t, x = tone(tone_char, duration, Fs)
        N = len(x)
        
        # Dominio del tempo
        ax_time = axes[idx, 0]
        ax_time.plot(t, x, linewidth=1)
        ax_time.set_title(f"Tono '{tone_char}' - Fs = {Fs} Hz")
        ax_time.set_xlabel('Tempo [s]')
        ax_time.set_ylabel('Ampiezza')
        ax_time.grid(True, alpha=0.3)
        
        # Dominio della frequenza
        ax_freq = axes[idx, 1]
        X = fft(x)
        X_shifted = fftshift(X)
        freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
        
        ax_freq.plot(freqs, np.abs(X_shifted), linewidth=1)
        ax_freq.set_title(f"Spettro - Fs = {Fs} Hz")
        ax_freq.set_xlabel('Frequenza [Hz]')
        ax_freq.set_ylabel('|X(f)|')
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(0, Fs/2)
        
        # Evidenzia aliasing se presente
        if Fs < 3300:
            ax_freq.axvline(x=Fs/2, color='red', linestyle='--', 
                          alpha=0.7, label='Nyquist limit')
            ax_freq.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    plt.show()
    plt.close()

def compare_sampling_frequencies_sequences_visual(seq='3456318060', duration, Fs_list, 
                                                 save_path=None):
    """
    Confronta visivamente una sequenza DTMF a diverse frequenze di campionamento.
    
    Parameters
    ----------
    seq : str
        Sequenza di tasti da analizzare
    duration : float
        Durata di ciascun tono nella sequenza
    Fs_list : list
        Lista di frequenze di campionamento da testare
    save_path : str
        Percorso per salvare il grafico
    """
    
    n_plots = len(Fs_list)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3*n_plots))
    
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, Fs in enumerate(Fs_list):
        # Genera la sequenza
        signal = dialNumber(seq, duration, Fs)
        N = len(signal)
        t = np.linspace(0, len(seq) * duration, N, endpoint=False)
        
        # Dominio del tempo
        ax_time = axes[idx, 0]
        ax_time.plot(t, signal, linewidth=1)
        ax_time.set_title(f"Sequenza '{seq}' - Fs = {Fs} Hz")
        ax_time.set_xlabel('Tempo [s]')
        ax_time.set_ylabel('Ampiezza')
        ax_time.grid(True, alpha=0.3)
        
        # Dominio della frequenza
        ax_freq = axes[idx, 1]
        
        X = fft(signal)
        X_shifted = fftshift(X)
        freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
        
        ax_freq.plot(freqs, np.abs(X_shifted), linewidth=1)
        ax_freq.set_title(f"Spettro - Fs = {Fs} Hz")
        ax_freq.set_xlabel('Frequenza [Hz]')
        ax_freq.set_ylabel('|X(f)|')
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(0, Fs/2)
        
        # Evidenzia aliasing se presente
        if Fs < 3300:
            ax_freq.axvline(x=Fs/2, color='red', linestyle='--', 
                           alpha=0.7, label='Nyquist limit')
            ax_freq.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.show()
    plt.close()


def demonstrate_aliasing(tone_char='3', duration=0.15, Fs_high=8000, Fs_low=2000, 
                        save_path=None):
    """
    Dimostra il fenomeno dell'aliasing confrontando campionamento diretto
    a frequenze diverse e l'effetto del filtro anti-alias durante ricampionamento.
    
    Parameters
    ----------
    tone_char : str
        Tasto da analizzare
    duration : float
        Durata del tono
    Fs_high : int
        Frequenza di campionamento alta (riferimento senza aliasing)
    Fs_low : int
        Frequenza di campionamento bassa (con aliasing)
    save_path : str
        Percorso per salvare il grafico
    """
    
    # Scenario 1: Campionamento diretto ad alta frequenza (riferimento)
    t_high, x_high = tone(tone_char, duration, Fs_high)
    
    # Scenario 2: Campionamento diretto a bassa frequenza (aliasing vero)
    t_low, x_low = tone(tone_char, duration, Fs_low)
    
    # Scenario 3: Ricampionamento con filtro anti-alias
    cutoff = 0.9 * (Fs_low / 2)
    x_high_filtered = lowpass_antialias_filter(x_high, Fs_high, cutoff)
    x_high_resampled = resample_signal(x_high_filtered, Fs_high, Fs_low)
    t_resampled = np.linspace(0, duration, len(x_high_resampled), endpoint=False)
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # ROW 1: Segnale ad alta Fs
    axes[0, 0].plot(t_high, x_high, linewidth=1)
    axes[0, 0].set_title(f"Campionamento diretto - Fs = {Fs_high} Hz (Riferimento)")
    axes[0, 0].set_xlabel('Tempo [s]')
    axes[0, 0].set_ylabel('Ampiezza')
    axes[0, 0].grid(True, alpha=0.3)
    
    X_high = fftshift(fft(x_high))
    freqs_high = np.linspace(-Fs_high/2, Fs_high/2, len(X_high), endpoint=False)
    axes[0, 1].plot(freqs_high, np.abs(X_high), linewidth=1)
    axes[0, 1].set_title(f"Spettro - Fs = {Fs_high} Hz")
    axes[0, 1].set_xlabel('Frequenza [Hz]')
    axes[0, 1].set_ylabel('|X(f)|')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, Fs_high/2)
    
    # ROW 2: Campionamento diretto a bassa Fs
    axes[1, 0].plot(t_low, x_low, linewidth=1, color='orange')
    axes[1, 0].set_title(f"Campionamento diretto - Fs = {Fs_low} Hz (Aliasing)")
    axes[1, 0].set_xlabel('Tempo [s]')
    axes[1, 0].set_ylabel('Ampiezza')
    axes[1, 0].grid(True, alpha=0.3)
    
    X_low = fftshift(fft(x_low))
    freqs_low = np.linspace(-Fs_low/2, Fs_low/2, len(X_low), endpoint=False)
    axes[1, 1].plot(freqs_low, np.abs(X_low), linewidth=1, color='orange')
    axes[1, 1].set_title(f"Spettro con aliasing - Fs = {Fs_low} Hz")
    axes[1, 1].set_xlabel('Frequenza [Hz]')
    axes[1, 1].set_ylabel('|X(f)|')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, Fs_low/2)
    axes[1, 1].axvline(x=Fs_low/2, color='red', linestyle='--', 
                       alpha=0.7, label='Nyquist limit')
    axes[1, 1].legend()
    
    # ROW 3: Ricampionamento con filtro anti-alias
    axes[2, 0].plot(t_resampled, x_high_resampled, linewidth=1, color='green')
    axes[2, 0].set_title(f"Ricampionamento con filtro anti-alias - Fs = {Fs_low} Hz")
    axes[2, 0].set_xlabel('Tempo [s]')
    axes[2, 0].set_ylabel('Ampiezza')
    axes[2, 0].grid(True, alpha=0.3)
    
    X_resampled = fftshift(fft(x_high_resampled))
    axes[2, 1].plot(freqs_low, np.abs(X_resampled), linewidth=1, color='green')
    axes[2, 1].set_title(f"Spettro filtrato (no aliasing) - Fs = {Fs_low} Hz")
    axes[2, 1].set_xlabel('Frequenza [Hz]')
    axes[2, 1].set_ylabel('|X(f)|')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim(0, Fs_low/2)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico aliasing salvato: {save_path}")
    
    plt.show()
    plt.close()


def test_accuracy_vs_sampling_frequency(Fs_list, duration=0.15, n_sequences=100):
    """
    Testa l'accuratezza di riconoscimento al variare di Fs.
    
    Parameters
    ----------
    Fs_list : list
        Lista di frequenze di campionamento da testare
    duration : float
        Durata di ciascun tono
    n_sequences : int
        Numero di sequenze casuali da testare
        
    Returns
    -------
    results : dict
        Dizionario con accuratezze per ogni Fs
    """
    results = {
        'Fs_list': Fs_list,
        'accuracies': [],
        'total_sequences': n_sequences
    }
    
    print("\nTEST ACCURATEZZA vs FREQUENZA DI CAMPIONAMENTO")
    
    for Fs in Fs_list:
        print(f"\nFs = {Fs} Hz")
        total = 0
        correct = 0
        
        for _ in range(n_sequences):
            seq = generate_random_sequence(10)
            true_seq = list(seq)
            
            signal = dialNumber(seq, duration, Fs)
            detected = recSequence(signal, duration, Fs)
            
            total += 1
            if detected == true_seq:
                correct += 1
        
        accuracy = 100 * correct / total
        results['accuracies'].append(accuracy)
        print(f"Accuratezza: {accuracy:.1f}% ({correct}/{total})")
    
    return results


def plot_accuracy_vs_fs(Fs_list, accuracies, save_path=None):
    """
    Grafico accuratezza vs frequenza di campionamento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(Fs_list, accuracies, 'o-', linewidth=2, markersize=8, color='#06A77D')
    plt.xlabel('Frequenza di campionamento Fs [Hz]', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Accuratezza di riconoscimento DTMF\nvs Frequenza di campionamento', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, 
                label='Accuratezza perfetta')
    plt.axvline(x=3266, color='red', linestyle='--', alpha=0.5, 
                label='Nyquist DTMF (2×1633)')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def run_sampling_experiments(Fs_list=None, duration=0.15, tone_char='9',
                             seq_example="3456318060", n_sequences=100,
                            plot_folder="Plots/experiments/sampling"):
    """
    Esegue tutti gli esperimenti sulla frequenza di campionamento.
    
    Genera:
    1. Confronti visivi segnale a diverse Fs
    2. Dimostrazione aliasing con/senza filtro
    3. Test accuratezza vs Fs
    4. Tabella risultati
    5. Grafico accuratezza vs Fs
    """
    if Fs_list is None:
        Fs_list = [2000, 3000, 4000, 6000, 8000, 16000]
    
    os.makedirs(plot_folder, exist_ok=True)
    
    print("ESPERIMENTI: FREQUENZA DI CAMPIONAMENTO")
    
    # Parte 1: Confronto visivo
    print("\n1. Generazione confronti visivi...")
    
    test_tones = ['9']
    for tone_char in test_tones:
        compare_sampling_frequencies_visual(tone_char, duration, Fs_list, save_path=os.path.join(plot_folder, f"sampling_comparison_{tone_char}.pdf"))

    # Parte 1b: Confronto visivo sequenza
    print("\n1b. Generazione confronto visivo sequenza...")
    compare_sampling_frequencies_sequences_visual(seq_example, duration, Fs_list,
    save_path=os.path.join(plot_folder, f"sampling_comparison_sequence.pdf"))
    
    # Parte 2: Dimostrazione aliasing
    print("\n2. Dimostrazione fenomeno aliasing...")
    demonstrate_aliasing(tone_char=tone_char, duration=duration, 
                        Fs_original=8000, Fs_target=2000, 
                        save_path=os.path.join(plot_folder, "aliasing_demonstration.pdf"))
    
    # Parte 3: Test accuratezza
    print("\n3. Test accuratezza vs frequenza di campionamento...")
    results = test_accuracy_vs_sampling_frequency(Fs_list, duration, n_sequences)
    
    # Grafico
    plot_accuracy_vs_fs(results['Fs_list'], results['accuracies'],
                       save_path=os.path.join(plot_folder, "accuracy_vs_fs.pdf"))
    
    # Tabella risultati
    print("\nTABELLA RISULTATI")
    print(f"{'Fs [Hz]':<10} {'Nyquist [Hz]':<15} {'Accuratezza [%]':<15} {'Note':<30}")
    print("-" * 70)
    
    for Fs, acc in zip(results['Fs_list'], results['accuracies']):
        nyquist = Fs / 2
        note = ""
        if Fs < 3266:
            note = "Sotto Nyquist DTMF - Aliasing"
        elif Fs < 4000:
            note = "Risoluzione insufficiente"
        else:
            note = "OK"
        
        print(f"{Fs:<10} {nyquist:<15.0f} {acc:<15.1f} {note:<30}")
    
    # Discussione aliasing
    print("\nDISCUSSIONE FENOMENO ALIASING")
    print("\nIl criterio di Nyquist richiede Fs >= 2 * f_max.")
    print(f"Per DTMF: f_max = 1633 Hz => Fs_min = 3266 Hz")
    print("\nRisultati osservati:")
    print("- Fs = 2000 Hz: Aliasing severo, accuratezza ~0%")
    print("- Fs = 3000 Hz: Nyquist soddisfatto ma risoluzione bassa")
    print("- Fs >= 4000 Hz: Riconoscimento affidabile")
    print("\nIl filtro anti-alias previene l'aliasing ma elimina")
    print("informazione utile quando Fs e' troppo bassa.")
    
    print(f"\nRisultati salvati in: {plot_folder}/")
    
    return results
