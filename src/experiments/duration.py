"""
Esperimenti per valutare l'effetto di durata del tono
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftshift
from sklearn.metrics import confusion_matrix
from src.DTMF_implementation import (
    tone,
    recNumber,
    dialNumber,
    recSequence,
    F1_DEFAULT,
    F2_DEFAULT,
    TONES_DEFAULT
)

# Costanti globali
ALL_TONES = ['1','2','3','4','5','6','7','8','9','0','*','#','A','B','C','D']

# FUNZIONI ATOMICHE - SINGOLI TONI

def plot_single_tone_multiple_durations(tone_char, durations=None, Fs=8000, 
                                        save_path=None):
    """
    Visualizza lo stesso tono DTMF a diverse durate nel dominio del tempo 
    e della frequenza per mostrare l'effetto della risoluzione spettrale.
    
    Parameters
    ----------
    tone_char : str
        Tasto DTMF da visualizzare
    durations : list, optional
        Liste di durate in secondi (default: [0.02, 0.05, 0.1, 0.2])
    Fs : int
        Frequenza di campionamento in Hz
    save_path : str, optional
        Percorso per salvare il grafico
    """
    if durations is None:
        durations = [0.02, 0.05, 0.1, 0.2]
    
    n_rows = len(durations)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5*n_rows))
    
    # Gestione caso singola durata
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Ottieni le frequenze del tono per annotazioni
    i, j = TONES_DEFAULT[tone_char]
    f1, f2 = F1_DEFAULT[i], F2_DEFAULT[j]
    
    for idx, duration in enumerate(durations):
        # Genera il tono
        t, x = tone(tone_char, duration, Fs)
        N = len(x)
        
        # Calcola risoluzione spettrale
        delta_f = 1.0 / duration
        
        # ===== PLOT DOMINIO DEL TEMPO =====
        ax_time = axes[idx, 0]
        ax_time.plot(t, x, linewidth=1.5, color='#2E86AB')
        ax_time.set_title(f"Tono '{tone_char}' - Durata: {duration}s\n"
                         f"({N} campioni)", 
                         fontsize=11)
        ax_time.set_xlabel('Tempo [s]', fontsize=10)
        ax_time.set_ylabel('Ampiezza', fontsize=10)
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xlim(0, duration)
        
        # ===== PLOT DOMINIO DELLA FREQUENZA =====
        ax_freq = axes[idx, 1]
        
        # Calcola FFT
        X = fft(x)
        X_shifted = fftshift(X)
        freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
        mag = np.abs(X_shifted)
        
        # Plot solo frequenze positive
        mask_pos = freqs >= 0
        freqs_pos = freqs[mask_pos]
        mag_pos = mag[mask_pos]
        
        ax_freq.plot(freqs_pos, mag_pos, linewidth=1.5, color='#E63946')
        ax_freq.set_title(f"Spettro - Risoluzione Δf = {delta_f:.1f} Hz", 
                         fontsize=11)
        ax_freq.set_xlabel('Frequenza [Hz]', fontsize=10)
        ax_freq.set_ylabel('|X(f)|', fontsize=10)
        ax_freq.grid(True, alpha=0.3)
        
        # Linee verticali alle frequenze teoriche
        ax_freq.axvline(x=f1, color='blue', linestyle='--', alpha=0.6, 
                       linewidth=1.5, label=f'F1 = {f1} Hz')
        ax_freq.axvline(x=f2, color='green', linestyle='--', alpha=0.6, 
                       linewidth=1.5, label=f'F2 = {f2} Hz')
        
        # Zoom sulle frequenze DTMF rilevanti
        ax_freq.set_xlim(500, 2000)
        ax_freq.legend(fontsize=9)
        
        # Annotazione critica per durate brevi
        if delta_f > 50:
            ax_freq.text(0.5, 0.95, 
                        f'⚠ Risoluzione insufficiente!\n'
                        f'Δf ({delta_f:.0f} Hz) > distanza min. tra F1 (73 Hz)',
                        transform=ax_freq.transAxes,
                        fontsize=9, verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Titolo generale
    fig.suptitle(f"Analisi tono DTMF '{tone_char}' al variare della durata\n"
                 f"Frequenze teoriche: {f1} Hz (F1) + {f2} Hz (F2)", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Salvataggio
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.show()
    plt.close()

def test_single_tones_at_duration(duration, Fs=8000, all_tones=None, n_trials=50):
    """
    Testa il riconoscimento di tutti i toni DTMF a una specifica durata
    con multiple realizzazioni per robustezza statistica.
    
    Parameters
    ----------
    duration : float
        Durata del tono in secondi
    Fs : int, optional
        Frequenza di campionamento (default: 8000)
    all_tones : list, optional
        Lista dei toni da testare (default: tutti i 16 toni)
    n_trials : int, optional
        Numero di test per ogni tono (default: 50)
    
    Returns
    -------
    results : dict
        Dizionario contenente:
        - 'accuracy': float, accuratezza in percentuale
        - 'correct': int, numero di riconoscimenti corretti
        - 'total': int, numero totale di toni testati
        - 'true_labels': list, etichette reali
        - 'predicted_labels': list, etichette predette
    """
    if all_tones is None:
        all_tones = ALL_TONES
    
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    
    print(f"\nTest durata: {duration}s (n_trials={n_trials} per tono)")
    
    # Contatore errori per tono (per report dettagliato)
    errors_per_tone = {}
    
    for tone_char in all_tones:
        tone_correct = 0
        
        # Testa ogni tono n_trials volte
        for trial in range(n_trials):
            # Genera il tono
            t, x = tone(tone_char, duration, Fs)
            
            # Riconosci
            detected = recNumber(x, Fs)
            
            # Registra risultati
            true_labels.append(tone_char)
            predicted_labels.append(detected if detected is not None else '?')
            
            if detected == tone_char:
                correct += 1
                tone_correct += 1
            
            total += 1
        
        # Calcola accuratezza per questo tono
        tone_accuracy = 100 * tone_correct / n_trials
        
        # Se ci sono errori, registra per report
        if tone_correct < n_trials:
            errors_per_tone[tone_char] = {
                'correct': tone_correct,
                'total': n_trials,
                'accuracy': tone_accuracy
            }
    
    accuracy = 100 * correct / total
    print(f"\nAccuratezza globale: {accuracy:.1f}% ({correct}/{total})")
    
    # Report errori se ce ne sono
    if errors_per_tone:
        print(f"\nToni con errori:")
        for tone_char, stats in sorted(errors_per_tone.items(), 
                                       key=lambda x: x[1]['accuracy']):
            print(f"  '{tone_char}': {stats['accuracy']:.1f}% "
                  f"({stats['correct']}/{stats['total']})")
    else:
        print("\nNessun errore - riconoscimento perfetto su tutti i toni!")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'errors_per_tone': errors_per_tone,
        'n_trials': n_trials
    }


def plot_accuracy_vs_duration_single(durations, accuracies, save_path=None):
    """
    Crea grafico dell'accuratezza vs durata per singoli toni.
    
    Parameters
    ----------
    durations : list
        Lista delle durate testate (in secondi)
    accuracies : list
        Lista delle accuratezze corrispondenti (in %)
    save_path : str, optional
        Percorso per salvare il grafico
    """

    plt.figure(figsize=(10, 6))
    plt.plot(durations, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Durata del tono [s]', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Accuratezza di riconoscimento di singoli toni DTMF\nal variare della durata', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Accuratezza perfetta')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
def plot_confusion_matrix(true_labels, predicted_labels, title, save_path=None):
    """
    Crea matrice di confusione con heatmap.
    
    Parameters
    ----------
    true_labels : list
        Etichette reali
    predicted_labels : list
        Etichette predette
    title : str
        Titolo del grafico
    save_path : str, optional
        Percorso per salvare il grafico
    """
    
    # Calcola matrice di confusione
    cm = confusion_matrix(true_labels, predicted_labels, labels=ALL_TONES)
    
    # Normalizza per riga (percentuale rispetto al totale per ogni classe reale)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=ALL_TONES, yticklabels=ALL_TONES,
                cbar_kws={'label': 'Frazione di predizioni'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Tono predetto', fontsize=12)
    plt.ylabel('Tono reale', fontsize=12)
    
    # Salvataggio
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close()
    
def analyze_errors(true_labels, predicted_labels, duration, F1=None, F2=None, tones=None):
    """
    Analizza e spiega gli errori più frequenti.
    
    Parameters
    ----------
    true_labels : list
        Etichette reali
    predicted_labels : list
        Etichette predette
    duration : float
        Durata del tono in secondi
    F1 : ndarray, optional
        Frequenze basse custom (default: ETSI standard)
    F2 : ndarray, optional
        Frequenze alte custom (default: ETSI standard)
    tones : dict, optional
        Dizionario custom dei toni (default: ETSI standard)
    
    Returns
    -------
    report : str
        Rapporto sugli errori analizzati
    """
    if F1 is None:
        F1 = F1_DEFAULT
    if F2 is None:
        F2 = F2_DEFAULT
    if tones is None:
        tones = TONES_DEFAULT
    
    # Conta gli errori
    errors = []
    for true_t, pred_t in zip(true_labels, predicted_labels):
        if true_t != pred_t:
            errors.append((true_t, pred_t))
    
    if not errors:
        return f"\nDurata {duration}s: Nessun errore - riconoscimento perfetto!\n"
    
    # Analizza frequenza degli errori
    error_counts = {}
    for true_t, pred_t in errors:
        key = (true_t, pred_t)
        error_counts[key] = error_counts.get(key, 0) + 1
    
    # Ordina per frequenza
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Genera report
    report = f"\n{'='*60}\n"
    report += f"ANALISI ERRORI - Durata: {duration}s\n"
    report += f"{'='*60}\n"
    report += f"Totale errori: {len(errors)}/{len(true_labels)} "
    report += f"({100*len(errors)/len(true_labels):.1f}%)\n\n"
    report += "Errori più frequenti:\n"
    
    for (true_t, pred_t), count in sorted_errors[:5]:  # Top 5
        report += f"\n  '{true_t}' → '{pred_t}': {count} volta/e\n"
        
        # Spiega il motivo dell'errore
        if true_t in tones and pred_t in tones:
            i_true, j_true = tones[true_t]
            i_pred, j_pred = tones[pred_t]
            
            report += f"Frequenze reali: {F1[i_true]:.0f} Hz, {F2[j_true]:.0f} Hz\n"
            report += f"Frequenze predette: {F1[i_pred]:.0f} Hz, {F2[j_pred]:.0f} Hz\n"
            
            # Diagnosi
            if i_true == i_pred:
                report += f"Confusione nella frequenza ALTA (stessa riga)\n"
            elif j_true == j_pred:
                report += f"Confusione nella frequenza BASSA (stessa colonna)\n"
            else:
                report += f"Confusione totale (riga e colonna diverse)\n"
        elif pred_t == '?':
            report += f"Tono non riconosciuto (risoluzione insufficiente)\n"
    
    # Spiegazione teorica
    resolution = 1.0 / duration
    report += f"\n{'='*60}\n"
    report += f"NOTA: Risoluzione spettrale Δf = 1/T = {resolution:.2f} Hz\n"
    report += f"La distanza minima tra frequenze DTMF è:\n"
    report += f"  - F1: {np.min(np.diff(F1)):.0f} Hz (es. 770-697 = 73 Hz)\n"
    report += f"  - F2: {np.min(np.diff(F2)):.0f} Hz (es. 1336-1209 = 127 Hz)\n"
    
    if resolution > 50:
        report += f"\nLa risoluzione di {resolution:.0f} Hz è INSUFFICIENTE per\n"
        report += f"distinguere frequenze DTMF vicine!\n"
    
    report += f"{'='*60}\n"
    
    return report
   
# FUNZIONI ATOMICHE - SEQUENZE
def plot_sequence_multiple_durations(seq, durations=None, Fs=8000, 
                                     save_path=None):
    """
    Visualizza una sequenza DTMF a diverse durate nel dominio del tempo 
    e della frequenza.
    
    Parameters
    ----------
    seq : str
        Sequenza di tasti DTMF
    durations : list, optional
        Liste di durate per tono in secondi (default: [0.05, 0.1, 0.2])
    Fs : int
        Frequenza di campionamento in Hz
    save_path : str, optional
        Percorso per salvare il grafico
    """
    if durations is None:
        durations = [0.05, 0.1, 0.2]
    
    n_rows = len(durations)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5*n_rows))
    
    # Gestione caso singola durata
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, duration in enumerate(durations):
        # Genera la sequenza
        signal = dialNumber(seq, duration, Fs)
        N = len(signal)
        total_duration = len(seq) * duration
        t = np.linspace(0, total_duration, N)
        
        # Calcola risoluzione spettrale
        delta_f = 1.0 / duration
        
        # ===== PLOT DOMINIO DEL TEMPO =====
        ax_time = axes[idx, 0]
        ax_time.plot(t, signal, linewidth=1, color='#2E86AB')
        ax_time.set_title(f"Sequenza '{seq}' - Durata per tono: {duration}s\n"
                         f"Durata totale: {total_duration}s ({N} campioni)", 
                         fontsize=11)
        ax_time.set_xlabel('Tempo [s]', fontsize=10)
        ax_time.set_ylabel('Ampiezza', fontsize=10)
        ax_time.grid(True, alpha=0.3)
        
        # Linee verticali per separare i toni
        for i in range(1, len(seq)):
            ax_time.axvline(x=i*duration, color='red', linestyle=':', 
                           alpha=0.4, linewidth=1)
        
        # Annotazioni dei tasti
        for i, char in enumerate(seq):
            ax_time.text((i + 0.5) * duration, ax_time.get_ylim()[1] * 0.9, 
                        f"'{char}'", ha='center', fontsize=10, 
                        fontweight='bold', color='darkred')
        
        # ===== PLOT DOMINIO DELLA FREQUENZA =====
        ax_freq = axes[idx, 1]
        
        # Calcola FFT
        X = fft(signal)
        X_shifted = fftshift(X)
        freqs = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
        mag = np.abs(X_shifted)
        
        # Plot solo frequenze positive
        mask_pos = freqs >= 0
        freqs_pos = freqs[mask_pos]
        mag_pos = mag[mask_pos]
        
        ax_freq.plot(freqs_pos, mag_pos, linewidth=1, color='#E63946')
        ax_freq.set_title(f"Spettro della sequenza - Risoluzione Δf = {delta_f:.1f} Hz", 
                         fontsize=11)
        ax_freq.set_xlabel('Frequenza [Hz]', fontsize=10)
        ax_freq.set_ylabel('|X(f)|', fontsize=10)
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(500, 2000)
        
        # Annota le frequenze presenti nella sequenza
        freq_set_f1 = set()
        freq_set_f2 = set()
        for char in seq:
            if char in TONES_DEFAULT:
                i, j = TONES_DEFAULT[char]
                freq_set_f1.add(F1_DEFAULT[i])
                freq_set_f2.add(F2_DEFAULT[j])
        
        # Linee verticali per frequenze presenti
        for f in sorted(freq_set_f1):
            ax_freq.axvline(x=f, color='blue', linestyle='--', alpha=0.4, linewidth=1)
        for f in sorted(freq_set_f2):
            ax_freq.axvline(x=f, color='green', linestyle='--', alpha=0.4, linewidth=1)
        
        # Legenda
        ax_freq.axvline(x=-1000, color='blue', linestyle='--', alpha=0.6, 
                       label='Freq. F1 presenti')
        ax_freq.axvline(x=-1000, color='green', linestyle='--', alpha=0.6, 
                       label='Freq. F2 presenti')
        ax_freq.legend(fontsize=9, loc='upper right')
        
        # Annotazione per risoluzione
        n_unique_tones = len(set(seq))
        ax_freq.text(0.02, 0.98, 
                    f'{n_unique_tones} toni unici nella sequenza\n'
                    f'{len(freq_set_f1)} freq. F1, {len(freq_set_f2)} freq. F2',
                    transform=ax_freq.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Titolo generale
    fig.suptitle(f"Analisi sequenza DTMF '{seq}' al variare della durata", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Salvataggio
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.show()
    plt.close()

def generate_random_sequence(length, all_tones=None):
    """
    Genera una sequenza casuale di tasti DTMF.
    
    Parameters
    ----------
    length : int
        Lunghezza della sequenza
    all_tones : list, optional
        Lista dei toni disponibili (default: tutti i 16 toni)
    
    Returns
    -------
    sequence : str
        Sequenza casuale di tasti
    """

    if all_tones is None:
        all_tones = ALL_TONES
    
    return ''.join(np.random.choice(all_tones, size=length))
    
def test_sequences_at_duration(duration, Fs=8000, n_sequences=50, seq_length=10):
    """
    Testa il riconoscimento di sequenze casuali a una specifica durata.
    
    Parameters
    ----------
    duration : float
        Durata di ciascun tono nella sequenza
    Fs : int, optional
        Frequenza di campionamento
    n_sequences : int, optional
        Numero di sequenze casuali da testare
    seq_length : int, optional
        Lunghezza di ciascuna sequenza
    
    Returns
    -------
    results : dict
        Dizionario contenente:
        - 'accuracy': float, % di sequenze completamente corrette
        - 'correct': int, numero di sequenze corrette
        - 'total': int, numero totale di sequenze testate
        - 'all_true_tones': list, tutti i toni reali (appiattiti)
        - 'all_predicted_tones': list, tutti i toni predetti (appiattiti)
    """
    correct_sequences = 0
    total_sequences = n_sequences
    
    all_true_tones = []
    all_predicted_tones = []
    
    print(f"\n--- Test sequenze (durata: {duration}s, n={n_sequences}) ---")
    
    for i in range(n_sequences):
        # Genera sequenza casuale
        seq = generate_random_sequence(seq_length)
        true_seq = list(seq)
        
        # Genera segnale
        signal = dialNumber(seq, duration, Fs)
        
        # Riconosci
        detected = recSequence(signal, duration, Fs)
        
        # Registra tutti i toni per matrice di confusione
        all_true_tones.extend(true_seq)
        all_predicted_tones.extend(detected if len(detected) == len(true_seq) else ['?']*len(true_seq))
        
        # Verifica se l'intera sequenza è corretta
        if detected == true_seq:
            correct_sequences += 1
    
    accuracy = 100 * correct_sequences / total_sequences
    print(f"Accuratezza: {accuracy:.1f}% ({correct_sequences}/{total_sequences} sequenze complete corrette)")
    
    return {
        'accuracy': accuracy,
        'correct': correct_sequences,
        'total': total_sequences,
        'all_true_tones': all_true_tones,
        'all_predicted_tones': all_predicted_tones
    }
  
def plot_accuracy_vs_duration_sequences(durations, accuracies, save_path=None):
    """
    Crea grafico dell'accuratezza vs durata per sequenze.
    
    Parameters
    ----------
    durations : list
        Lista delle durate testate (in secondi)
    accuracies : list
        Lista delle accuratezze corrispondenti (in %)
    save_path : str, optional
        Percorso per salvare il grafico
    """

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(durations, accuracies, 's-', linewidth=2, markersize=8, color='#A23B72')
    plt.xlabel('Durata del tono [s]', fontsize=12)
    plt.ylabel('Accuratezza [%]', fontsize=12)
    plt.title('Accuratezza di riconoscimento di sequenze DTMF\nal variare della durata', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Accuratezza perfetta')
    plt.legend()
    
    # Salvataggio
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Grafico salvato: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()
   
# WRAPPER

def run_duration_experiments(durations=None, Fs=8000, n_sequences=100,
                            n_trials=50, tone_char='9', seq='3456318060',
                            plot_folder="Plots/improvements/duration"):
    """
    Esegue tutti gli esperimenti sulla variazione della durata del tono.
    
    Genera:
    - Grafico singolo tono a diverse durate (tempo + frequenza)
    - Grafico sequenza a diverse durate (tempo + frequenza)
    - Grafico accuratezza vs durata (singoli toni)
    - Matrici di confusione per durate critiche
    - Report errori testuali
    - Grafico accuratezza vs durata (sequenze)
    - Matrice di confusione per sequenze (durate critiche)
    
    Parameters
    ----------
    durations : list, optional
        Durate da testare in secondi (default: [0.02, 0.05, 0.1, 0.15, 0.2, 0.5])
    Fs : int
        Frequenza di campionamento
    n_sequences : int
        Numero di sequenze casuali da testare per ogni durata
    n_trials : int
        Numero di test per ogni tono nei test su singoli toni (default: 50)
    tone_char : str, optional
        Tasto per visualizzazione singolo tono (default: '9')
    seq : str, optional
        Sequenza per visualizzazione (default: '3456318060')
    plot_folder : str
        Cartella dove salvare i grafici
    
    Returns
    -------
    results : dict
        Dizionario con tutti i risultati degli esperimenti
    """
    if durations is None:
        durations = [0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
    
    os.makedirs(plot_folder, exist_ok=True)

    print("\nVisualizzazione effetto durata su segnale e spettro...")
    
    # Singolo tono a diverse durate
    durations_visual = [0.02, 0.05, 0.1, 0.2]  # Subset per visualizzazione
    plot_single_tone_multiple_durations(
        tone_char=tone_char,
        durations=durations_visual,
        Fs=Fs,
        save_path=os.path.join(plot_folder, "single_tone_durations_comparison.pdf")
    )
    
    # Sequenza a diverse durate
    plot_sequence_multiple_durations(
        seq=seq,
        durations=[0.05, 0.1, 0.2],
        Fs=Fs,
        save_path=os.path.join(plot_folder, "sequence_durations_comparison.pdf")
    )
    
    print("ESPERIMENTI: VARIAZIONE DELLA DURATA DEL TONO")
    print("PARTE 1: TEST SU SINGOLI TONI")
    
    single_tone_results = {}
    single_accuracies = []
    
    for duration in durations:
        result = test_single_tones_at_duration(duration, Fs, n_trials=n_trials)
        single_tone_results[duration] = result
        single_accuracies.append(result['accuracy'])
    
    # Grafico accuratezza vs durata (singoli toni)
    plot_accuracy_vs_duration_single(
        durations, 
        single_accuracies,
        save_path=os.path.join(plot_folder, "accuracy_vs_duration_single_tones.pdf")
    )
    
    # Matrici di confusione per durate critiche
    critical_durations = [d for d in durations if d <= 0.1]  # Durate più critiche
    
    for duration in critical_durations:
        result = single_tone_results[duration]
        plot_confusion_matrix(
            result['true_labels'],
            result['predicted_labels'],
            f'Matrice di Confusione - Singoli Toni\nDurata: {duration}s',
            save_path=os.path.join(plot_folder, f"confusion_matrix_single_{duration}s.pdf")
        )
    
    # Analisi errori e salvataggio report
    report_path = os.path.join(plot_folder, "error_analysis_single_tones.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ANALISI ERRORI - TEST SINGOLI TONI\n")
        f.write("="*60 + "\n\n")
        
        for duration in durations:
            result = single_tone_results[duration]
            error_report = analyze_errors(
                result['true_labels'],
                result['predicted_labels'],
                duration
            )
            f.write(error_report)
            print(error_report)
    
    print(f"\nReport errori salvato: {report_path}")
    print("PARTE 2: TEST SU SEQUENZE")
    
    sequence_results = {}
    sequence_accuracies = []
    
    np.random.seed(42)  # Per riproducibilità
    
    for duration in durations:
        result = test_sequences_at_duration(duration, Fs, n_sequences)
        sequence_results[duration] = result
        sequence_accuracies.append(result['accuracy'])
    
    # Grafico accuratezza vs durata (sequenze)
    plot_accuracy_vs_duration_sequences(
        durations,
        sequence_accuracies,
        save_path=os.path.join(plot_folder, "accuracy_vs_duration_sequences.pdf")
    )
    
    # Matrici di confusione per durate critiche (sequenze)
    for duration in critical_durations:
        result = sequence_results[duration]
        plot_confusion_matrix(
            result['all_true_tones'],
            result['all_predicted_tones'],
            f'Matrice di Confusione - Sequenze\nDurata: {duration}s',
            save_path=os.path.join(plot_folder, f"confusion_matrix_sequences_{duration}s.pdf")
        )
    
    print("\nRIEPILOGO RISULTATI")

    print("\nAccuratezza singoli toni:")
    for duration, acc in zip(durations, single_accuracies):
        result = single_tone_results[duration]
        n_trials = result.get('n_trials', 1)
        total_tests = 16 * n_trials
        print(f"  {duration}s: {acc:.1f}% (su {total_tests} test)")
    
    print("\nAccuratezza sequenze:")
    for duration, acc in zip(durations, sequence_accuracies):
        print(f"  {duration}s: {acc:.1f}%")
    
    print(f"\nTutti i risultati salvati in: {plot_folder}/")
    
    return {
        'single_tones': single_tone_results,
        'sequences': sequence_results,
        'durations': durations
    }

   