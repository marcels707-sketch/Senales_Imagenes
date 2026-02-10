from scipy.io import wavfile as waves
import pyaudio as pa 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import lfilter, firwin, firwin2, remez, minimum_phase
import wave


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


#--------FUNCIONES DE FILTROS FIR--------
#---------------- Ventanas ----------------
def fir_bandpass(fc1, fc2, fs, numtaps=201, window='hamming'):
    nyq = fs / 2
    return firwin(numtaps, [fc1/nyq, fc2/nyq], window=window, pass_zero=False)

# ---------------- KAISER ----------------
def fir_bandpass_kaiser(fc1, fc2, fs, numtaps=301, beta=8.6):
    nyq = fs / 2
    return firwin(numtaps, [fc1/nyq, fc2/nyq], window=('kaiser', beta), pass_zero=False)

# ---------------- Parks–McClellan ----------------
def fir_bandpass_remez(fc1, fc2, fs, numtaps=151):
    nyq = fs / 2.0

    # Bordes en Hz (no normalizados)
    f_stop1 = fc1 * 0.8
    f_pass1 = fc1
    f_pass2 = fc2
    f_stop2 = fc2 * 1.2

    # Asegurar que no se pasen del Nyquist
    f_stop2 = min(f_stop2, nyq * 0.999)

    # Bandas en Hz, con fs explícito
    bands = [0.0,
             f_stop1,
             f_pass1,
             f_pass2,
             f_stop2,
             nyq]

    # 3 bandas: [0–f_stop1], [f_pass1–f_pass2], [f_stop2–nyq]
    desired = [0, 1, 0]
    weights = [1, 1, 1]

    b = remez(numtaps, bands, desired, weight=weights, fs=fs)
    return b

# ---------------- Fase minima ----------------
def fir_minphase_bandpass(fc1, fc2, fs, numtaps=301):
    b = fir_bandpass(fc1, fc2, fs, numtaps=numtaps)
    return minimum_phase(b)

# ----------- Respuesta arbitraria ------------
def fir_custom(freqs, gains, fs, numtaps=301):
    freqs_norm = np.array(freqs) / (fs/2)
    return firwin2(numtaps, freqs_norm, gains)


##############PROGRAMA PRINCIPAL##############
#Cargar el archivo de datos (propio o cargado)
filename = seleccionar_datos()
if filename == 'q':
    exit()
Fs, Audio_m, data = cargar_datos(filename)
reproducir_simple(Audio_m,Fs,'original')

#normalizada de los datos
data_norm = data /np.max(np.abs(data))

#Parametros de longitud y frecuencias para imprimir
L = len(data_norm)
fft_original = np.fft.fft(data_norm)
frecuencias = np.fft.fftfreq(L, 1/Fs)

#Datos de frecuencia utilizados en el rango
fc_1 = 300
fc_2 = 1500

# FIR estándar
b = fir_bandpass(fc_1, fc_2, Fs, numtaps=301, window='hamming')
a = 1
ruta = f'Laboratorios/Tarea2/data/fir_hamming_{fc_1}-{fc_2}.wav'
final, fft_hamming, sig_hamming = guarda_archivo(b, a, data_norm, ruta)
reproducir_simple(final, Fs, 'Hamming')

# FIR Kaiser
b = fir_bandpass_kaiser(fc_1, fc_2, Fs, beta=8.6)
a = 1
ruta = f'Laboratorios/Tarea2/data/fir_kaiser_{fc_1}-{fc_2}.wav'
final, fft_kaiser, sig_kaiser = guarda_archivo(b, a, data_norm, ruta)
reproducir_simple(final, Fs, 'Kaiser')

# FIR Remez
b = fir_bandpass_remez(fc_1, fc_2, Fs)
a = 1
ruta = f'Laboratorios/Tarea2/data/fir_remez_{fc_1}-{fc_2}.wav'
final, fft_remez, sig_remez = guarda_archivo(b, a, data_norm, ruta)
reproducir_simple(final, Fs, 'Remez')

# FIR MINPHASE
b = fir_minphase_bandpass(fc_1, fc_2, Fs)
a = 1
ruta = f'Laboratorios/Tarea2/data/fir_remez_{fc_1}-{fc_2}.wav'
final, fft_phase, sig_phase = guarda_archivo(b, a, data_norm, ruta)
reproducir_simple(final, Fs, 'Phase')

# FIR CUSTOMS
b = fir_custom(
    freqs=[0, fc_1, fc_1, fc_2, fc_2, Fs/2],
    gains=[0, 0, 1, 1, 0, 0],
    fs=Fs
)
a = 1
ruta = f'Laboratorios/Tarea2/data/fir_remez_{fc_1}-{fc_2}.wav'
final, fft_custom, sig_custom = guarda_archivo(b, a, data_norm, ruta)
reproducir_simple(final, Fs, 'Custom')

ffts_dict = {
    "Hamming": fft_hamming,
    "Kaiser": fft_kaiser,
    "Remez": fft_remez,
    "Phase": fft_phase,
    "Custom": fft_custom
}

sign_dict = {
    "Hamming": sig_hamming,
    "Kaiser": sig_kaiser,
    "Remez": sig_remez,
    "Phase": sig_phase,
    "Custom": sig_custom
}

graficar_subplots_completos(frecuencias,fft_original,ffts_dict,data_norm,sign_dict,Fs)
graficar_comparativa_completa(frecuencias,fft_original,ffts_dict,data_norm,sign_dict,Fs)