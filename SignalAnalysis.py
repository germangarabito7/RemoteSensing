import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft




# Definimos los parámetros de las señales
T = 1.0  # periodo
fs = 500  # frecuencia de muestreo
t = np.linspace(0, T, int(2*T), endpoint=False)  # vector de tiempo

# Generamos la señal chirp
f0 = 1e9
f1 = 1.5e9
chirp_signal = chirp(t, f0, T, f1, method='linear')

# Generamos la señal BPSK
np.random.seed(0)
bits = np.random.randint(0, 2, int(2*T))
print(bits)
bpsk_signal = np.cos(2*np.pi*5*t + np.pi*bits)

# Graficamos las señales en el dominio del tiempo
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, chirp_signal)
plt.title('Señal Chirp')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 2)
plt.plot(t, bpsk_signal)
plt.title('Señal BPSK')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()

# Calculamos el espectro de las señales
chirp_spectrum = np.abs(fft(chirp_signal))
bpsk_spectrum = np.abs(fft(bpsk_signal))

# Graficamos el espectro de las señales
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(chirp_spectrum)
plt.title('Espectro de la señal Chirp')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 2)
plt.plot(bpsk_spectrum)
plt.title('Espectro de la señal BPSK')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()
