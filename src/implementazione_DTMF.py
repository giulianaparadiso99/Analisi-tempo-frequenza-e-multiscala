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

def save_plot(filename, folder="Plots"):
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    print(f"Plot salvato in: {os.path.join(folder, filename)}")

def tone(number, duration, Fs, tones=None, F1=None, F2=None):
    """Genera il tono DTMF per un singolo tasto,
    generando un errore se il tasto non è valido"""

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

def tone_plot_save(t, x, Fs, filename="tono.wav", plot_filename="tono.png"):
    """Plot e salvataggio del segnale + salvataggio file audio"""
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(t, x)
    plt.title('Segnale DTMF')
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ampiezza")
    plt.grid(True)
    save_plot(plot_filename)
    plt.show()
    plt.close()

    # Salvataggio WAV
    x_norm = x / np.max(np.abs(x))
    x_int16 = np.int16(x_norm * 32767)
    write(filename, Fs, x_int16)
    print(f"Audio salvato: {filename}")

def dialNumber(numbers, toneDuration, Fs):
    """Genera una sequenza concatenata di toni DTMF"""
    full_signal = np.array([])

    for i in numbers:
        _, x = tone(i, toneDuration, Fs)
        full_signal = np.concatenate((full_signal, x))

    return full_signal

def plot_and_save_sequence(signal, Fs, filename="sequenza.wav", plot_filename="sequenza.png"):
    """Visualizza la sequenza DTMF nel tempo e salva WAV + PNG"""

    # Asse dei tempi
    t = np.linspace(0, len(signal)/Fs, len(signal))

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, signal)
    plt.title("Sequenza DTMF nel dominio del tempo")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ampiezza")
    plt.grid(True)
    save_plot(plot_filename)
    plt.show()
    plt.close()

    # Audio
    signal_norm = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal_norm * 32767)
    write(filename, Fs, signal_int16)
    print(f"Audio salvato: {filename}")

def plot_spectrum(signal, Fs, plot_filename="spettro.png", save_folder="Plots"):
    """
    Calcola e visualizza lo spettro e salva il plot nella cartella desiderata.
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

    # Salvataggio
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, plot_filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot salvato in: {path}")

    plt.show()
    plt.close()

    return freqs, X_shifted

def recNumber(x, Fs, F1=None, F2=None, tones=None):
    """
    Riconosce un singolo tono DTMF dal suo segnale x(t).
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
    Riconosce una sequenza di toni concatenati.
    """
    samples_per_tone = int(Fs * toneDuration)
    N = len(signal)

    detected = []

    # Processa ogni blocco
    for start in range(0, N, samples_per_tone):
        end = start + samples_per_tone
        x = signal[start:end]

        if len(x) < samples_per_tone:
            break  # evita ultimo spezzone troppo corto

        key = recNumber(x, Fs)
        detected.append(key)

    return detected

def dialNumber_pauses(numbers, toneDuration, Fs, pauseDuration=0.05):
    """
    Genera una sequenza di toni DTMF inserendo una pausa di silenzio
    tra un tono e l'altro.
    """
    full_signal = np.array([])
    pause = np.zeros(int(Fs * pauseDuration))

    for i in numbers:
        _, x = tone(i, toneDuration, Fs)
        full_signal = np.concatenate((full_signal, x, pause))

    return full_signal

def detect_sequence_pauses(signal, toneDuration, pauseDuration, Fs, silence_threshold=0.05):
    """
    Riconosce una sequenza di toni DTMF concatenati con pause di silenzio.
    I segmenti di pausa vengono ignorati.
    """

    # segmentazione: tono + pausa
    block_size = int(Fs * (toneDuration + pauseDuration))
    N = len(signal)

    detected = []

    for start in range(0, N, block_size):
        end = start + block_size
        x = signal[start:end]

        if len(x) < block_size:
            break

        # estrazione dei toni
        tone_samples = int(Fs * toneDuration)
        x_tone = x[:tone_samples]

        # le pause vengono saltate
        if np.max(np.abs(x_tone)) < silence_threshold:
            continue

        # riconoscimento
        key = recNumber(x_tone, Fs)
        detected.append(key)

    return detected

