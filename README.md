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

```bash
pip install -r requirements.txt
```

---

## Moduli Implementati

### 1. DTMF_implementation.py

Modulo base contenente le funzioni fondamentali:

**Generazione toni:**
- `tone(number, duration, Fs)` - Genera un singolo tono DTMF
- `dialNumber(numbers, toneDuration, Fs)` - Genera una sequenza di toni
- `dialNumber_pauses(...)` - Genera sequenza con pause

**Riconoscimento:**
- `recNumber(x, Fs)` - Riconosce un singolo tono
- `recSequence(signal, toneDuration, Fs)` - Riconosce una sequenza
- `detect_sequence_pauses(...)` - Riconosce sequenza con pause

**Visualizzazione:**
- `plot_spectrum(signal, Fs)` - Visualizza lo spettro di frequenza
- `tone_plot_save(...)` - Salva tono come file WAV e visualizza
- `plot_all_tones_grid(...)` - Visualizza griglia 4×4 di tutti i toni

### 2. Esperimenti

#### experiments/duration.py

Analisi dell'effetto della durata del tono:
- Test su singoli toni (tutti i 16 caratteri, 50 prove ciascuno)
- Test su sequenze casuali (100 sequenze da 10 toni)
- Matrici di confusione a diverse durate critiche
- Identificazione soglia minima di riconoscimento

#### experiments/noise.py

Analisi dell'effetto del rumore gaussiano:
- Aggiunta rumore gaussiano bianco con diversi σ
- Calcolo SNR (Signal-to-Noise Ratio)
- Test accuratezza vs σ e vs SNR
- Heatmap durata × rumore (1000 sequenze per configurazione)
- Visualizzazioni zoom temporale e spettrale

#### experiments/sampling.py

Analisi dell'effetto della frequenza di campionamento:
- Generazione diretta a diverse Fs (non resampling)
- Dimostrazione fenomeno aliasing
- Filtri anti-aliasing preventivi
- Test accuratezza vs Fs (100 sequenze per valore)

### 3. Miglioramenti

#### improvements/zero_padding.py

Implementazione e analisi zero-padding:
- `detect_single_tone_with_zeropadding(...)` - Riconoscimento con padding
- `recSequence_with_zeropadding(...)` - Riconoscimento sequenze con padding
- Heatmap durata × fattore padding
- Effetto sulla risoluzione spettrale apparente

#### improvements/denoising.py

Tecniche di denoising (1000 sequenze per σ):
- `moving_average_denoise(...)` - Filtro a media mobile (M=5)
- `leaky_integrator_denoise(...)` - Leaky integrator (λ=0.8)
- `bandpass_filter_dtmf(...)` - Filtro passa-banda [650-1700 Hz]
- Confronti visuali tempo/frequenza per σ = {0.2, 0.5, 1.0}

---

## Utilizzo

### 1. Generazione di un tono singolo

```python
from src.DTMF_implementation import tone, tone_plot_save

# Genera il tono '5' di 0.15 secondi
Fs = 8000
t, x = tone('5', duration=0.15, Fs=Fs)

# Visualizza e salva
tone_plot_save(t, x, Fs, filename="tono_5.wav")
```

### 2. Generazione e riconoscimento di una sequenza

```python
from src.DTMF_implementation import dialNumber, recSequence

# Genera sequenza
sequenza = "3456318060"
signal = dialNumber(sequenza, toneDuration=0.15, Fs=8000)

# Riconosci
detected = recSequence(signal, toneDuration=0.15, Fs=8000)
print(f"Sequenza riconosciuta: {''.join(detected)}")
```

### 3. Esecuzione esperimenti completi

```python
# Esperimenti sulla durata
from src.experiments.duration import run_duration_experiments
results_duration = run_duration_experiments(
    durations=durations_list,
    Fs=8000,
    plot_folder="Plots/experiments/duration"
)

# Esperimenti sul rumore
from src.experiments.noise import run_noise_experiments
results_noise = run_noise_experiments(
    sigmas=[0, 0.5, 1, 2, 3, 4, 5, 6],
    duration=0.15,
    Fs=8000,
    plot_folder="Plots/experiments/noise"
)

# Esperimenti sulla frequenza di campionamento
from src.experiments.sampling import run_sampling_experiments
results_fs = run_sampling_experiments(
    Fs_list=[2000, 3000, 3266, 4000, 8000, 16000],
    duration=0.15,
    plot_folder="Plots/experiments/sampling"
)
```

---

## Risultati Principali

### Effetto della Durata

- **Soglia critica:** T = 18 ms (Δf = 56 Hz) per accuratezza 100%
- **Punto di svolta:** T = 11.1 ms (50% accuratezza singoli toni)
- **Durata < 11 ms:** Accuratezza < 20% (collasso spettrale)
- **Durata ≥ 18 ms:** Accuratezza 100% stabile
- **Pattern errori:** Confusione sistematica tra toni stessa colonna (stessa F2) per durate borderline

### Effetto del Rumore

- **σ ≤ 2 (SNR ≥ -6 dB):** Accuratezza ~100% (sistema intrinsecamente robusto)
- **σ = 3 (SNR ≈ -9.5 dB):** Accuratezza ~93% (inizio degrado)
- **σ = 4 (SNR ≈ -12 dB):** Accuratezza ~58% (soglia critica)
- **σ ≥ 5:** Accuratezza < 25% (riconoscimento inaffidabile)
- **Analisi congiunta durata-rumore:** Fattori fortemente accoppiati, durate lunghe migliorano SNR effettivo

### Effetto della Frequenza di Campionamento

- **Fs = 2000 Hz:** Accuratezza 0% (aliasing di 1477 Hz → 523 Hz)
- **Fs = 3000 Hz:** Accuratezza 7% (sotto Nyquist DTMF)
- **Fs = 3266 Hz:** Accuratezza 100% (esattamente 2 × 1633 Hz, limite teorico Nyquist)
- **Fs ≥ 4000 Hz:** Accuratezza 100% stabile
- **Transizione netta:** Da 7% a 100% in soli 266 Hz (3000 → 3266)

### Zero-Padding

- **Zona 1 (T < 4 ms):** 0% indipendentemente dal padding (limite fisico informazione)
- **Zona 2 (4 ≤ T < 12 ms):** Zero-padding critico
  - T = 5 ms: 0% (1×) → 100% (4×)
  - T = 6 ms: 8% (1×) → 29% (2×) → 100% (4×)
  - T = 7 ms: 0% (1×) → 100% (2×)
- **Zona 3 (T ≥ 12 ms):** 100% anche senza padding
- **Riduzione durata minima:** 58% (da 12 ms a 5 ms con padding 4×)

### Tecniche di Denoising

- **Media mobile (M=5):** Controproducente, degrada a 48% già con σ=1 (attenua 1477 Hz)
- **Leaky integrator (λ=0.8):** Identico al baseline fino a σ=2, miglioramento marginale per σ≥3
- **Filtro passa-banda [650-1700 Hz]:** Qualitativamente superiore, elimina completamente rumore fuori banda
- **Nessun filtro:** Sorprendentemente robusto fino a σ=2 (100% accuratezza)

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
3. Identifica le due frequenze dominanti cercando i picchi massimi nelle bande F1 e F2
4. Mappa la coppia (f_low, f_high) al tasto corrispondente tramite dizionario