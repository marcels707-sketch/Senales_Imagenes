from scipy.io import wavfile as waves
import pyaudio as pa 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, lfilter
from scipy.signal import cheby1, cheby2, ellip, bessel
import wave
import scipy.fftpack as fourier


def grabar_voz(filename):
    FRAMES = 1024       # resolucion sonido
    FORMAT = pa.paInt16  # enteros de 16 bits
    CHANNELS = 1    
    Fs = 44100
    segundos = 5

    p = pa.PyAudio()

    # Abrir stream de grabación
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=Fs,
                input=True,
                frames_per_buffer=FRAMES)


    frames_totales = []
 
    # Calculamos cuántos paquetes leer para completar 3 segundos
    for i in range(0, int(Fs / FRAMES * segundos)):
        data = stream.read(FRAMES)
        frames_totales.append(data)
 
    print("Grabación finalizada.")
 
    # Detener y cerrar stream
    stream.stop_stream()
    stream.close()
    p.terminate()
 
    # --- GUARDAR ARCHIVO .WAV ---
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(Fs)
    wf.writeframes(b''.join(frames_totales))
    wf.close()
    print(f"Archivo guardado como: {filename}")
    return filename

def seleccionar_datos():
    
    while True:
        print("--------------------------------")
        print("Ingrese las siguientes opciones:")
        print("--------------------------------")
        print("1 Grabar un nuevo audio---------")
        print("2 Audio de ejemplo")  
        print("q Salir")
        seleccion = input("Ingrese su selección: ")

        # Opcion de escalada
        if seleccion == '1':
            filename = 'Laboratorios/Tarea2/data/grabacion_nueva.wav'
            grabar_voz(filename)
            break
        elif seleccion == '2':
            filename = 'Laboratorios/Tarea2/data/grabacion_2.wav'
            break
        elif seleccion == 'q':
            filename = 'q'
            break
        else:
            print("Error no se encuentra la opción ingresada "+seleccion)
    return filename    

#Funcion que permite cargar la informacion del archivo de sonido
def cargar_datos(filename):
    # leer el archivo y encontrar sus frecuencias y componentes
    Fs, data = waves.read(filename) 
    # comprobar que hay datos
    if len(data.shape) > 1: 
        # [componentes, canal]
        Audio_m = data [:,0] 
    else:
        Audio_m = data
    #retorna la frecuencia y sus componentes 
    return Fs, Audio_m, data

#Funcion que permite reproducir el sonido seleccionado
def reproducir_simple(data, fs, tipo):
    print("Reproduciendo audio "+tipo)
    sd.play(data, fs)
    sd.wait()  # Espera a que termine la reproducción
    print("-------Fin de la reproducción-------")

#Funcion que guarda la informacion de los filtros
def guarda_archivo (b,a,data_norm,ruta):
    audio_filtrado_iir = lfilter (b,a,data_norm)

    fft_filtrada = np.fft.fft(audio_filtrado_iir)
    # escalar a 16 bits
    audio_final = (audio_filtrado_iir*32767).astype(np.int16)
    #guarda el archivo en la carpeta de datos
    waves.write(ruta,Fs,audio_final)
    #retorna audio_fnal
    return audio_final, fft_filtrada, audio_filtrado_iir

#Funcion que permite dibujar las respectuvas frecuencias resultantes
def graficar_subplots_completos(frecuencias, fft_original, ffts_dict, data_norm, seniales_dict, Fs):
    filtros = list(ffts_dict.keys())
    total = len(filtros)
    t = np.arange(len(data_norm)) / Fs

    plt.figure(figsize=(26, 8))

    for i, nombre in enumerate(filtros):
        L = len(fft_original)

        # -------- FILA 1: FFT --------
        plt.subplot(2, total, i+1)
        plt.plot(frecuencias[:L//2], np.abs(fft_original[:L//2]), alpha=0.4, label="Original")
        plt.plot(frecuencias[:L//2], np.abs(ffts_dict[nombre][:L//2]), label=nombre)
        plt.title(f"{nombre} - Frecuencia")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Magnitud")
        plt.grid()
        plt.legend(fontsize=8)
        plt.xlim(0, 1000)

        # -------- FILA 2: TIEMPO --------
        plt.subplot(2, total, total + i + 1)
        plt.plot(t, data_norm, alpha=0.4, label="Original")
        plt.plot(t, seniales_dict[nombre], label="Filtrado")
        plt.title(f"{nombre} - Tiempo")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud")
        plt.grid()
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


#grafica comparativa en una sola ventana
def graficar_comparativa_completa(frecuencias, fft_original, ffts_dict, data_norm, seniales_dict, Fs):
    L = len(fft_original)
    t = np.arange(len(data_norm)) / Fs

    plt.figure(figsize=(16, 10))

    # -------- FILA 1: FFT --------
    plt.subplot(2, 1, 1)
    plt.plot(frecuencias[:L//2], np.abs(fft_original[:L//2]), label="Original", linewidth=2)

    for nombre, fft_f in ffts_dict.items():
        plt.plot(frecuencias[:L//2], np.abs(fft_f[:L//2]), label=nombre)

    plt.title("Comparativa de todos los filtros - Frecuencia")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid()
    plt.legend()
    plt.xlim(0, 1000)

    # -------- FILA 2: TIEMPO --------
    plt.subplot(2, 1, 2)
    plt.plot(t, data_norm, alpha=0.4, label="Original", linewidth=2)

    for nombre, sig in seniales_dict.items():
        plt.plot(t, sig, label=nombre)

    plt.title("Comparativa de todos los filtros - Tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


#--------FUNCIONES DE FILTROS IIR--------
#---------------- BUTTER ----------------
def butter_lowpass(fc,fs,order=5): # order = 2-5
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    b , a = butter(order, normal_fc, btype='low', analog=False)
    return b,a

def butter_highpass(fc,fs,order=5): # order = 2-5
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    b , a = butter(order, normal_fc, btype='high', analog=False)
    return b,a  

def butter_bandpass(fc_1,fc_2,fs,order=5): # order = 2-5
    nyq = 0.5 * fs
    low = fc_1 / nyq
    high = fc_2 / nyq
    b , a = butter(order,[low,high], btype='band', analog=False)
    return b,a  

# ---------------- CHEBYSHEV I ----------------
def cheby1_lowpass(fc, fs, rp=1, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return cheby1(order, rp, normal_fc, btype='low')

def cheby1_highpass(fc, fs, rp=1, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return cheby1(order, rp, normal_fc, btype='high')

def cheby1_bandpass(fc1, fc2, fs, rp=1, order=5):
    nyq = 0.5 * fs
    low = fc1 / nyq
    high = fc2 / nyq
    return cheby1(order, rp, [low, high], btype='band')

# ---------------- CHEBYSHEV II ----------------
def cheby2_lowpass(fc, fs, rs=50, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return cheby2(order, rs, normal_fc, btype='low')

def cheby2_highpass(fc, fs, rs=50, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return cheby2(order, rs, normal_fc, btype='high')

def cheby2_bandpass(fc1, fc2, fs, rs=50, order=5):
    nyq = 0.5 * fs
    low = fc1 / nyq
    high = fc2 / nyq
    return cheby2(order, rs, [low, high], btype='band')


# ---------------- ELÍPTICO ----------------
def ellip_lowpass(fc, fs, rp=1, rs=50, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return ellip(order, rp, rs, normal_fc, btype='low')

def ellip_highpass(fc, fs, rp=1, rs=50, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return ellip(order, rp, rs, normal_fc, btype='high')

def ellip_bandpass(fc1, fc2, fs, rp=1, rs=50, order=5):
    nyq = 0.5 * fs
    low = fc1 / nyq
    high = fc2 / nyq
    return ellip(order, rp, rs, [low, high], btype='band')


# ---------------- BESSEL ----------------
def bessel_lowpass(fc, fs, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return bessel(order, normal_fc, btype='low', norm='phase')

def bessel_highpass(fc, fs, order=5):
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    return bessel(order, normal_fc, btype='high', norm='phase')

def bessel_bandpass(fc1, fc2, fs, order=5):
    nyq = 0.5 * fs
    low = fc1 / nyq
    high = fc2 / nyq
    return bessel(order, [low, high], btype='band', norm='phase')


##############PROGRAMA PRINCIPAL##############
#Cargar el archivo de datos (propio o cargado)
filename = seleccionar_datos()
if filename == 'q':
    exit()
Fs, Audio_m, data = cargar_datos(filename)
reproducir_simple(Audio_m,Fs,'original')

L = len(Audio_m)
n = np.arange(0, L) / Fs

fft = fourier.fft(Audio_m)
M_fft = np.abs(fft[:L//2])
F = Fs * np.arange(0, L//2) / L

plt.figure(figsize=(10,6))

# --- Subplot 1: señal en el tiempo ---
plt.subplot(2,1,1)
plt.plot(n, Audio_m)
plt.title('Audio con ruido blanco')
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

# --- Subplot 2: magnitud de la FFT ---
plt.subplot(2,1,2)
plt.plot(F, M_fft)
plt.xlim(0, 4000)
plt.grid(True)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')

plt.tight_layout()
plt.show()


#normalizada de los datos
data_norm = data /np.max(np.abs(data))

#Parametros de longitud y frecuencias para imprimir
L = len(data_norm)
fft_original = np.fft.fft(data_norm)
frecuencias = np.fft.fftfreq(L, 1/Fs)



#Datos de frecuencia utilizados en el rango
fc_1 = 300
fc_2 = 1500

#Aplicacion FIltro de butter
b,a = butter_bandpass(fc_1,fc_2,Fs)
ruta = 'Laboratorios/Tarea2/data/butter_pb_'+str(fc_1)+'-'+str(fc_2)+'.wav'
final,fft_butter,sig_butter = guarda_archivo (b,a,data_norm,ruta)
reproducir_simple(final,Fs, 'butter')

#Aplicacion Filtro de Chebyshev1
b,a = cheby1_bandpass(fc_1,fc_2,Fs)
ruta = 'Laboratorios/Tarea2/data/cheby1_pb_'+str(fc_1)+'-'+str(fc_2)+'.wav'
final,fft_cheby1,sig_cheby1 = guarda_archivo (b,a,data_norm,ruta)
reproducir_simple(final,Fs, 'chebyshev1')

#Aplicacion Filtro de Chebyshev2
b,a = cheby2_bandpass(fc_1,fc_2,Fs)
ruta = 'Laboratorios/Tarea2/data/cheby2_pb_'+str(fc_1)+'-'+str(fc_2)+'.wav'
final,fft_cheby2,sig_cheby2 = guarda_archivo (b,a,data_norm,ruta)
reproducir_simple(final,Fs,'chebyshev2')

#Aplicacion Filtro Eliptico
b,a = ellip_bandpass(fc_1,fc_2,Fs)
ruta = 'Laboratorios/Tarea2/data/ellip_pb_'+str(fc_1)+'-'+str(fc_2)+'.wav'
final, fft_elip, sig_elip = guarda_archivo (b,a,data_norm,ruta)
reproducir_simple(final,Fs,'eliptico')

#Aplicacion Filtro Bessel
b,a = bessel_bandpass(fc_1,fc_2,Fs)
ruta = 'Laboratorios/Tarea2/data/bessel_pb_'+str(fc_1)+'-'+str(fc_2)+'.wav'
final,fft_bessel,sig_bessel = guarda_archivo (b,a,data_norm,ruta)
reproducir_simple(final,Fs,'bessel')


ffts_dict = {
    "Butterworth": fft_butter,
    "Chebyshev I": fft_cheby1,
    "Chebyshev II": fft_cheby2,
    "Elíptico": fft_elip,
    "Bessel": fft_bessel
}

sign_dict = {
    "Butterworth": sig_butter,
    "Chebyshev I": sig_cheby1,
    "Chebyshev II": sig_cheby2,
    "Elíptico": sig_elip,
    "Bessel": sig_bessel
}

graficar_subplots_completos(frecuencias,fft_original,ffts_dict,data_norm,sign_dict,Fs)
graficar_comparativa_completa(frecuencias,fft_original,ffts_dict,data_norm,sign_dict,Fs)