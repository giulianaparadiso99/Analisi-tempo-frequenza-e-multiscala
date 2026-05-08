# Analisi Tempo-Frequenza e Multiscala

Homework DTMF - Generazione e riconoscimento di tonalità Dual-Tone Multi-Frequency

**Autrice:** Giuliana Paradiso  
**Corso:** Analisi Tempo-Frequenza e Multiscala  
**Istituto:** Politecnico di Torino  
**Anno Accademico:** 2025/2026

---

## Descrizione del Progetto

Questo progetto implementa un sistema completo per la generazione e il riconoscimento di toni DTMF (Dual-Tone Multi-Frequency), la tecnologia utilizzata nei telefoni per la trasmissione di numeri e simboli attraverso coppie di tonalità sinusoidali.

### Obiettivi

- Generare toni DTMF associati a singoli tasti e sequenze
- Riconoscere sequenze di toni tramite analisi spettrale (FFT)
- Analizzare sperimentalmente l'effetto di diverse variabili:
  - Durata del tono
  - Presenza di rumore gaussiano
  - Frequenza di campionamento
- Implementare tecniche di miglioramento (zero-padding, filtraggio)

---

## Struttura del Progetto

Analisi-tempo-frequenza-e-multiscala/
├── src/
│   ├── experiments/
│   │   ├── duration.py
│   │   ├── noise.py
│   │   └── sampling.py
│   ├── improvements/
│   │   ├── denoising.py
│   │   └── zero_padding.py
│   ├── DTMF_implementation.py
│   └── generazione-e-riconoscimento-di-tonalit-dtmf.ipynb
├── report/
│   └── Homework_analisi_tempo_frequenza.pdf
├── README.md
└── requirements.txt

---

## Requisiti

### Dipendenze Python

numpy
matplotlib
scipy
scikit-learn
seaborn
jupyter

Installazione:

pip install -r requirements.txt

---

## Moduli Implementati

### 1. DTMF_implementation.py

Modulo base contenente le funzioni fondamentali:

**Generazione toni:**
- tone(number, duration, Fs) - Genera un singolo tono DTMF
- dialNumber(numbers, toneDuration, Fs) - Genera una sequenza di toni
- dialNumber_pauses(...) - Genera sequenza con pause

**Riconoscimento:**
- recNumber(x, Fs) - Riconosce un singolo tono
- recSequence(signal, toneDuration, Fs) - Riconosce una sequenza

**Visualizzazione:**
- plot_spectrum(signal, Fs) - Visualizza lo spettro di frequenza
- tone_plot_save(...) - Salva tono come file WAV

### 2. Esperimenti

#### experiments/noise.py

Analisi dell'effetto del rumore gaussiano:
- Calcolo SNR (Signal-to-Noise Ratio)
- Test accuratezza vs σ e vs SNR
- Matrice durata × rumore
- Visualizzazioni tempo/frequenza

#### experiments/duration.py

Analisi dell'effetto della durata del tono:
- Test su singoli toni (tutti i 16 caratteri)
- Test su sequenze casuali
- Matrici di confusione
- Analisi errori più frequenti

#### experiments/sampling.py

Analisi dell'effetto della frequenza di campionamento:
- Sottocampionamento e aliasing
- Filtri anti-aliasing
- Test accuratezza vs Fs

### 3. Miglioramenti

#### improvements/denoising.py

Tecniche di denoising:
- Filtro a media mobile
- Leaky integrator
- Filtro passa-banda DTMF (650-1700 Hz)

#### improvements/zero_padding.py

Implementazione e analisi zero-padding:
- Effetto sulla risoluzione spettrale
- Test su toni di breve durata

---

## Utilizzo

### 1. Generazione di un tono singolo

from src.DTMF_implementation import tone, tone_plot_save

# Genera il tono '5' di 0.15 secondi
Fs = 8000
t, x = tone('5', duration=0.15, Fs=Fs)

# Visualizza e salva
tone_plot_save(t, x, Fs, filename="tono_5.wav")

### 2. Generazione e riconoscimento di una sequenza

from src.DTMF_implementation import dialNumber, recSequence

# Genera sequenza
sequenza = "3456318060"
signal = dialNumber(sequenza, toneDuration=0.15, Fs=8000)

# Riconosci
detected = recSequence(signal, toneDuration=0.15, Fs=8000)
print(f"Sequenza riconosciuta: {''.join(detected)}")

### 3. Esecuzione esperimenti completi

# Esperimenti sul rumore
from experiments.noise import run_noise_experiments
results_noise = run_noise_experiments(save_folder="risultati/noise")

# Esperimenti sulla durata
from experiments.duration import run_duration_experiments
results_duration = run_duration_experiments(save_folder="risultati/duration")

# Esperimenti sulla frequenza di campionamento
from experiments.sampling import run_sampling_experiments
results_fs = run_sampling_experiments(save_folder="risultati/sampling")

---

## Risultati Principali

### Effetto della Durata

- Durata < 0.05s: Accuratezza degrada rapidamente
- Durata ≥ 0.1s: Accuratezza 100%
- Risoluzione spettrale: Δf = 1/T limita la separabilità delle frequenze DTMF vicine

### Effetto del Rumore

- σ ≤ 1: Accuratezza > 99%
- σ = 2: Accuratezza ~90%
- σ ≥ 4: Accuratezza < 20%
- SNR > 10 dB: Riconoscimento affidabile

### Effetto della Frequenza di Campionamento

- Fs = 2000 Hz: Accuratezza ~0% (aliasing)
- Fs = 3000 Hz: Accuratezza ~6% (risoluzione insufficiente)
- Fs ≥ 4000 Hz: Accuratezza 100%
- Criterio di Nyquist: Necessario Fs ≥ 2 × 1633 Hz = 3266 Hz

### Tecniche di Miglioramento

- Media mobile: Peggiora l'accuratezza (attenua anche i toni DTMF)
- Leaky integrator: Lieve miglioramento per σ elevati
- Filtro passa-banda: Soluzione più efficace per rumore fuori banda
- Zero-padding: Migliora visualizzazione ma non la risoluzione reale

---

## Riferimenti

- ETSI TS 101 235-1 V1.1.1 - "Specification of dual tone multi-frequency (DTMF) transmitters and receivers; Part 1: General"
- ETSI TS 101 235-3 V1.1.1 - "Specification of dual tone multi-frequency (DTMF) transmitters and receivers; Part 3: Receivers"

---

## Note Tecniche

### Standard DTMF

Il sistema DTMF utilizza una matrice 4×4 di frequenze:

|        | 1209 Hz | 1336 Hz | 1477 Hz | 1633 Hz |
|--------|---------|---------|---------|---------|
| 697 Hz |    1    |    2    |    3    |    A    |
| 770 Hz |    4    |    5    |    6    |    B    |
| 852 Hz |    7    |    8    |    9    |    C    |
| 941 Hz |    *    |    0    |    #    |    D    |

Ogni tasto corrisponde alla somma di due sinusoidi:

x(t) = cos(2πf_low·t) + cos(2πf_high·t)

### Riconoscimento via FFT

L'algoritmo di riconoscimento:
1. Segmenta il segnale in blocchi di durata fissa
2. Applica la FFT a ciascun blocco
3. Identifica le due frequenze dominanti
4. Mappa la coppia (f_low, f_high) al tasto corrispondente