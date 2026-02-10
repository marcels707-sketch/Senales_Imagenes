from scipy.io import wavfile as waves
import pyaudio as pa 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, lfilter, firwin
import scipy.fftpack as fourier


#Funcion que permite reproducir el sonido seleccionado
def reproducir_simple(data, fs, tipo):
    print("Reproduciendo audio "+tipo)
    sd.play(data, fs)
    sd.wait()  # Espera a que termine la reproducción
    print("-------Fin de la reproducción-------")


def butter_highpass(fc,fs,order=5): # order = 2-5
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    b , a = butter(order, normal_fc, btype='high', analog=False)
    return b,a 

filename = 'Laboratorios/Tarea2/data/Idles.wav'

Fs, data = waves.read(filename) # leer el archivo y encontrar sus frecuencias y componentes

if len(data.shape) > 1: # comprobar que hay datos
    Audio_m = data [:,0] # [componentes, canal]
else:
    Audio_m = data

L = len(Audio_m)

fft = fourier.fft(Audio_m)   # realizar la FFT
M_fft = abs(fft)   # Magnitud de la señal
M_fft = M_fft[0:L//2]   # Mitad de los datos

F = Fs*np.arange(0,L//2)/L 
n = np.arange(0,L)/Fs # teorema de muestreo

plt.figure(figsize=(8,4))
plt.plot(F,M_fft)
plt.xlim(0,1000)
plt.ylabel('Magnitud')
plt.xlabel('Frecuencias Hz')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(n,Audio_m)
plt.title('Nota SOL en el tiempo')
plt.grid(True)
plt.show()

data_norm = data /np.max(np.abs(data))
L = len(data_norm)
fft = np.fft.fft(data_norm)
frecuencias = np.fft.fftfreq(L,1/Fs)


#### filtro pasa alto
fc = 2000

fc_hp = 22000  # frecuencia de corte
numtaps = 301

b_hp = firwin(numtaps=numtaps, cutoff=fc_hp, fs=Fs, pass_zero=False)
a_hp = 1

audio_filtrado = lfilter(b_hp, a_hp, data_norm)

audio_filtrado = audio_filtrado / np.max(np.abs(audio_filtrado))

audio_final = (audio_filtrado * 32767).astype(np.int16)
waves.write("Laboratorios/Tarea2/data/pasa_altos.wav", Fs, audio_final)
#reproducir_simple(audio_final, Fs, "Pasa Altos FIR")








#b,a = butter_highpass(fc,Fs,order= 5)
# aplicar el filtro
#audio_filtrado_iir = lfilter (b,a,data_norm)

#fft_filtrada = np.fft.fft(audio_filtrado_iir)
# escalar a 16 bits
#audio_final = (audio_filtrado_iir*32767).astype(np.int16)
#ruta = 'Laboratorios/Tarea2/data/ResultadoIdles.wav'
#waves.write(ruta,Fs,audio_final)
