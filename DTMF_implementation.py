"""
Implementazione base del sistema DTMF
- Generazione toni singoli e sequenze
- Riconoscimento via FFT
- Visualizzazione e salvataggio
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.io.wavfile import write

# Costanti globali
F1_DEFAULT = np.array([697, 770, 852, 941])
F2_DEFAULT = np.array([1209, 1336, 1477, 1633])

TONES_DEFAULT = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2), 'A': (0, 3),
    '4': (1, 0), '5': (1, 1), '6': (1, 2), 'B': (1, 3),
    '7': (2, 0), '8': (2, 1), '9': (2, 2), 'C': (2, 3),
    '*': (3, 0), '0': (3, 1), '#': (3, 2), 'D': (3, 3)
}


def tone(number, duration, Fs, tones=None, F1=None, F2=None):
    """Genera il tono DTMF per un singolo tasto,
    generando un errore se il tasto non è valido.

    Parameters
    ----------
    number : str
        Tasto da generare
    duration : float
        Durata del tono in secondi
    Fs : int
        Frequenza di campionamento in Hz
    tones : dict, optional
        Dizionario custom dei toni (default: ETSI standard)
    F1 : ndarray, optional
        Frequenze basse custom (default: ETSI standard)
    F2 : ndarray, optional
        Frequenze alte custom (default: ETSI standard)
        
    Returns
    -------
    t : ndarray
        Array dei tempi
    x : ndarray
        Segnale DTMF generato
        
    Raises
    ------
    KeyError
        Se il tasto non è valido
    """
    # Usa valori default se non specificati
    if tones is None:
        tones = TONES_DEFAULT
    if F1 is None:
        F1 = F1_DEFAULT
    if F2 is None:
        F2 = F2_DEFAULT

    if number not in tones:
        raise KeyError(f" Il tasto '{number}' non esiste nella tastiera DTMF")
    
    i, j = tones[number]
    f1, f2 = F1[i], F2[j] 
    
    t = np.linspace(0, duration, int(Fs * duration))
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
    return t, x

def tone_plot_save(t, x, Fs, filename="tono.wav", plot_filename="tono.pdf", plot_folder="Plots/DTMF_implementation", audio_folder="Audio/tones"):
    """
    Visualizza e salva un singolo tono DTMF.
    
    Parameters
    ----------
    t : ndarray
        Array dei tempi
    x : ndarray
        Segnale DTMF
    Fs : int
        Frequenza di campionamento
    filename : str, optional
        Nome del file audio WAV
    plot_filename : str, optional
        Nome del file immagine PDF
    plot_folder : str, optional
        Cartella di salvataggio per i plot
    audio_folder : str, optional
        Cartella di salvataggio per i file audio
    """

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(t, x)
    plt.title('Segnale DTMF')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ampiezza")
    plt.grid(True)

    # Salvataggio plot
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', bbox_inches="tight")
    print(f"Plot salvato in: {os.path.join(plot_folder, plot_filename)}")
    plt.show()
    plt.close()

    # Salvataggio WAV
    os.makedirs(audio_folder, exist_ok=True)
    audio_path = os.path.join(audio_folder, filename)
    x_norm = x / np.max(np.abs(x))
    x_int16 = np.int16(x_norm * 32767)
    write(audio_path, Fs, x_int16)
    print(f"Audio salvato: {audio_path}")

def plot_all_tones_grid(duration=0.15, Fs=8000, plot_folder="Plots/DTMF_implementation"):
    """
    Visualizza tutti i 16 toni DTMF in una griglia 4x4.
    Genera due figure: dominio del tempo e dominio della frequenza.
    """
    keypad_rows = [
        ['1', '2', '3', 'A'],
        ['4', '5', '6', 'B'],
        ['7', '8', '9', 'C'],
        ['*', '0', '#', 'D']
    ]
    
    # Plot tempo
    fig_time, axes_time = plt.subplots(4, 4, figsize=(20, 16))
    fig_time.suptitle('Tutti i toni DTMF nel dominio del tempo', 
                     fontsize=20, fontweight='bold', y=0.995)
    for r in range(4):
        for c in range(4):
            key = keypad_rows[r][c]
            t, x = tone(key, duration, Fs)
            ax = axes_time[r, c]
            ax.plot(t, x)
            ax.set_title(f"Tasto '{key}'", fontsize=12)
            ax.set_xlabel("Tempo [s]")
            ax.set_ylabel("Ampiezza")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    time_path = os.path.join(plot_folder, "all_tones_time.pdf")
    fig_time.savefig(time_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot frequenza
    fig_freq, axes_freq = plt.subplots(4, 4, figsize=(20, 16))
    fig_freq.suptitle('Tutti i toni DTMF nel dominio della frequenza', 
                     fontsize=20, fontweight='bold', y=0.995)
    for r in range(4):
        for c in range(4):
            key = keypad_rows[r][c]
            t, x = tone(key, duration, Fs)
            N = len(x)
            X = fftshift(fft(x))
            freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
            ax = axes_freq[r, c]
            ax.plot(freqs, np.abs(X))
            ax.set_title(f"Spettro '{key}'", fontsize=12)
            ax.set_xlabel("Frequenza [Hz]")
            ax.set_ylabel("|X(f)|")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, Fs/2)
    
    plt.tight_layout()
    freq_path = os.path.join(plot_folder, "all_tones_freq.pdf")
    fig_freq.savefig(freq_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Plot salvati in: {plot_folder}/")

def dialNumber(numbers, toneDuration, Fs):
    """
    Genera una sequenza concatenata di toni DTMF.
    
    Parameters
    ----------
    numbers : str
        Sequenza di tasti da generare (es. "3456318060")
    toneDuration : float
        Durata di ciascun tono in secondi
    Fs : int
        Frequenza di campionamento in Hz
        
    Returns
    -------
    full_signal : ndarray
        Segnale completo della sequenza
    """

    full_signal = np.array([])

    for i in numbers:
        _, x = tone(i, toneDuration, Fs)
        full_signal = np.concatenate((full_signal, x))

    return full_signal

def plot_and_save_sequence(signal, Fs, filename="sequenza.wav", plot_filename="sequenza.pdf", plot_folder="Plots/DTMF_implementation", audio_folder="Audio/sequences"):
    """
    Visualizza e salva una sequenza di toni DTMF.
    
    Parameters
    ----------
    signal : ndarray
        Segnale della sequenza
    Fs : int
        Frequenza di campionamento in Hz
    filename : str, optional
        Nome del file audio WAV
    plot_filename : str, optional
        Nome del file immagine PDF
    plot_folder : str, optional
        Cartella di salvataggio per i plot
    audio_folder : str, optional
        Cartella di salvataggio per i file audio
    """

    # Asse dei tempi
    t = np.linspace(0, len(signal)/Fs, len(signal))

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, signal)
    plt.title("Sequenza DTMF nel dominio del tempo")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ampiezza")
    plt.grid(True)

    # Salvataggio plot
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', bbox_inches="tight")
    print(f"Plot salvato in: {plot_path}")
    plt.show()
    plt.close()

    # Audio
    os.makedirs(audio_folder, exist_ok=True)
    audio_path = os.path.join(audio_folder, filename)
    signal_norm = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal_norm * 32767)
    write(audio_path, Fs, signal_int16)
    print(f"Audio salvato: {audio_path}")

def plot_spectrum(signal, Fs, plot_filename="spettro.pdf", plot_folder="Plots/DTMF_implementation",
                 annotate_dtmf=False, sequence=None):
    """
    Calcola e visualizza lo spettro di un segnale DTMF.
    
    Parameters
    ----------
    signal : ndarray
        Segnale da analizzare
    Fs : int
        Frequenza di campionamento
    plot_filename : str, optional
        Nome del file immagine
    plot_folder : str, optional
        Cartella di salvataggio
    annotate_dtmf : bool, optional
        Se True, aggiunge linee verticali alle frequenze DTMF teoriche
    sequence : str, optional
        Sequenza di tasti (es. "3456318060") per annotare solo le frequenze presenti
        
    Returns
    -------
    freqs : ndarray
        Array delle frequenze
    X_shifted : ndarray
        Trasformata di Fourier centrata
    """

    N = len(signal)

    # DFT
    X = fft(signal)
    X_shifted = fftshift(X)

    # Asse delle frequenze
    freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(freqs, np.abs(X_shifted))
    plt.title("Spettro DTMF")
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel("|X(f)|")
    plt.grid(True)
    
    # Annotazioni frequenze DTMF
    if annotate_dtmf:
        if sequence is not None:
            # Trova solo le frequenze presenti nella sequenza
            freq_set = set()
            for char in sequence:
                if char in TONES_DEFAULT:
                    i, j = TONES_DEFAULT[char]
                    freq_set.add(F1_DEFAULT[i])
                    freq_set.add(F2_DEFAULT[j])
            
            # Disegna linee per le frequenze trovate
            for f in sorted(freq_set):
                plt.axvline(x=f, color='red', linestyle='--', alpha=0.4, linewidth=1)
                plt.axvline(x=-f, color='red', linestyle='--', alpha=0.4, linewidth=1)
        else:
            # Disegna tutte le frequenze DTMF
            for f in F1_DEFAULT:
                plt.axvline(x=f, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
                plt.axvline(x=-f, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
            
            for f in F2_DEFAULT:
                plt.axvline(x=f, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
                plt.axvline(x=-f, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
            
            # Legenda
            plt.axvline(x=-5000, color='red', linestyle='--', alpha=0.5, 
                       linewidth=1, label='F1 (697-941 Hz)')
            plt.axvline(x=-5000, color='blue', linestyle='--', alpha=0.5, 
                       linewidth=1, label='F2 (1209-1633 Hz)')
            plt.legend(loc='upper right')

    # Salvataggio
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Plot salvato in: {plot_path}")

    plt.show()
    plt.close()

    return freqs, X_shifted

def recNumber(x, Fs, F1=None, F2=None, tones=None):
    """
    Riconosce un singolo tono DTMF dal suo segnale.
    
    Parameters
    ----------
    x : ndarray
        Segnale del tono
    Fs : int
        Frequenza di campionamento in Hz
    F1 : ndarray, optional
        Frequenze basse custom (default: ETSI standard)
    F2 : ndarray, optional
        Frequenze alte custom (default: ETSI standard)
    tones : dict, optional
        Dizionario custom dei toni (default: ETSI standard)

    Returns
    -------
    key : str or None
        Tasto riconosciuto, None se non riconosciuto
    """

    # Usa valori default se non specificati
    if tones is None:
        tones = TONES_DEFAULT
    if F1 is None:
        F1 = F1_DEFAULT
    if F2 is None:
        F2 = F2_DEFAULT

    N = len(x)

    # FFT
    X = fft(x)
    X_shifted = fftshift(X)
    freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)

    mag = np.abs(X_shifted)

    # Si considerano solo frequenze positive
    mask_pos = freqs >= 0
    freqs_pos = freqs[mask_pos]
    mag_pos = mag[mask_pos]

    # Si sceglie la riga (frequenza bassa) tra F1
    amps_F1 = []
    for f in F1:
        idx = np.argmin(np.abs(freqs_pos - f))
        amps_F1.append(mag_pos[idx])
    i = int(np.argmax(amps_F1))

    # Si sceglie la colonna (frequenza alta) tra F2
    amps_F2 = []
    for f in F2:
        idx = np.argmin(np.abs(freqs_pos - f))
        amps_F2.append(mag_pos[idx])
    j = int(np.argmax(amps_F2))

    # Si mappa (i,j) nel dizionario dei toni
    for key, (r, c) in tones.items():
        if r == i and c == j:
            return key

    return None

def recSequence(signal, toneDuration, Fs):
    """
    Riconosce una sequenza di toni DTMF concatenati.
    
    Parameters
    ----------
    signal : ndarray
        Segnale della sequenza
    toneDuration : float
        Durata di ciascun tono in secondi
    Fs : int
        Frequenza di campionamento
        
    Returns
    -------
    detected : list
        Lista dei tasti riconosciuti
    """
    samples_per_tone = int(Fs * toneDuration)
    N = len(signal)

    detected = []

    # Processa ogni blocco
    for start in range(0, N, samples_per_tone):
        end = start + samples_per_tone
        x = signal[start:end]

        if len(x) < samples_per_tone:
            break 

        key = recNumber(x, Fs)
        detected.append(key)

    return detected

def dialNumber_pauses(numbers, toneDuration, Fs, pauseDuration=0.05):
    """
    Genera una sequenza di toni DTMF con pause di silenzio.
    
    Parameters
    ----------
    numbers : str
        Sequenza di tasti da generare
    toneDuration : float
        Durata di ciascun tono in secondi
    Fs : int
        Frequenza di campionamento in Hz
    pauseDuration : float, optional
        Durata della pausa tra toni in secondi (default: 0.05)
        
    Returns
    -------
    full_signal : ndarray
        Segnale completo con pause
    """

    full_signal = np.array([])
    pause = np.zeros(int(Fs * pauseDuration))

    for i in numbers:
        _, x = tone(i, toneDuration, Fs)
        full_signal = np.concatenate((full_signal, x, pause))

    return full_signal

def detect_sequence_pauses(signal, toneDuration, pauseDuration, Fs, silence_threshold=0.05):
    """
    Riconosce una sequenza DTMF con pause, ignorando i silenzi.
    
    Parameters
    ----------
    signal : ndarray
        Segnale della sequenza con pause
    toneDuration : float
        Durata di ciascun tono
    pauseDuration : float
        Durata delle pause
    Fs : int
        Frequenza di campionamento
    silence_threshold : float, optional
        Soglia per rilevare il silenzio
        
    Returns
    -------
    detected : list
        Lista dei tasti riconosciuti
    """

    # Segmentazione: tono + pausa
    block_size = int(Fs * (toneDuration + pauseDuration))
    N = len(signal)

    detected = []

    for start in range(0, N, block_size):
        end = start + block_size
        x = signal[start:end]

        if len(x) < block_size:
            break

        # Estrazione dei toni
        tone_samples = int(Fs * toneDuration)
        x_tone = x[:tone_samples]

        # Le pause vengono saltate
        if np.max(np.abs(x_tone)) < silence_threshold:
            continue

        # Riconoscimento
        key = recNumber(x_tone, Fs)
        detected.append(key)

    return detected

