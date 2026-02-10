from scipy.io import wavfile as waves
import numpy as np
from scipy.signal import butter, cheby2, filtfilt, remez, lfilter, firwin

filename = 'Laboratorios/Tarea2/data/Idles.wav'
Fs, data = waves.read(filename)

Audio_m = data[:,0] if len(data.shape) > 1 else data
L = len(Audio_m)

# ============================
# BUTTERWORTH HPF
# ============================

fc = 3000
order = 5
nyq = 0.5 * Fs
normal_fc = fc / nyq

b_butt, a_butt = butter(order, normal_fc, btype='high')
audio_butt = filtfilt(b_butt, a_butt, Audio_m)

audio_butt /= np.max(np.abs(audio_butt))
waves.write('Laboratorios/Tarea2/data/Idles_HP_Butterworth.wav',
            Fs, (audio_butt*32767).astype(np.int16))

# ============================
# CHEBYSHEV II HPF
# ============================

rs = 40
b_cheb, a_cheb = cheby2(order, rs, normal_fc, btype='high')
audio_cheb = filtfilt(b_cheb, a_cheb, Audio_m)

audio_cheb /= np.max(np.abs(audio_cheb))
waves.write('Laboratorios/Tarea2/data/Idles_HP_Cheby2.wav',
            Fs, (audio_cheb*32767).astype(np.int16))

# ============================
# FIR PARKS–MCCLELLAN HPF
# ============================

trans = 50
bands = [0, fc-trans, fc, Fs/2]
desired = [0, 1]
weights = [1, 1]
numtaps = 801   # <<<<< mucho más razonable

b_fir = remez(numtaps, bands, desired, weight=weights, fs=Fs)

# usar lfilter, no filtfilt
audio_fir = lfilter(b_fir, [1], Audio_m)

max_val = np.max(np.abs(audio_fir))
if max_val > 1e-6:
    audio_fir /= max_val

waves.write('Laboratorios/Tarea2/data/Idles_HP_FIR_PM.wav',
            Fs, (audio_fir*32767).astype(np.int16))


#low_cut = 300
#high_cut = 1100

low_cut = 1000
high_cut = 2000

nyq = 0.5 * Fs

# Convertir a frecuencias normalizadas
low = low_cut / nyq
high = high_cut / nyq

# Filtro Butterworth Pasa-Banda
b, a = butter(order, [low, high], btype='band')
audio_vocal = filtfilt(b, a, Audio_m)

# Normalización y guardado
audio_vocal /= np.max(np.abs(audio_vocal))
waves.write('Laboratorios/Tarea2/data/Idles_Voz_Filtro.wav', 
            Fs, (audio_vocal*32767).astype(np.int16))

rs = 40
b_cheb, a_cheb = cheby2(order, rs, normal_fc, btype='high')
audio_cheb = filtfilt(b_cheb, a_cheb, Audio_m)

audio_cheb /= np.max(np.abs(audio_cheb))
waves.write('Laboratorios/Tarea2/data/Idles_Cheby2.wav',
            Fs, (audio_cheb*32767).astype(np.int16))

b_ham = firwin(numtaps, [low, high], window='hamming', pass_zero=False)
audio_banda = filtfilt(b_ham, 1, Audio_m)

waves.write('Laboratorios/Tarea2/data/Idles_banda.wav',
            Fs, (audio_banda*32767).astype(np.int16))