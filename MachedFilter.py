from FunctionsRadar import *
import numpy as np
import matplotlib.pyplot as plt
import time

def SignalAnalysis(x,y,fs,SignalName=None):
	plt.figure(f"Signal {SignalName}")
	plt.subplot(211)
	plt.plot(x, y, linewidth = 0.7)
	plt.title(f'Time Domain of {SignalName} Signal')
	plt.xlabel('Time(s)')
	plt.ylabel('Voltage(V)')
	plt.grid()

	# Compute and Plot Chirp Signal in Frequency Domain
	# Compute FFT
	y_fft_PSD = np.fft.fftshift(np.abs(np.fft.fft(y))**2/len(y))
	x_fft_PSD = np.fft.fftshift(np.fft.fftfreq(len(y),1/fs))

	plt.subplot(212)
	plt.plot(x_fft_PSD,y_fft_PSD,linewidth = 0.7)
	plt.title(f'Frequency Domain of {SignalName} Signal')
	plt.xlabel('Frequency(Hz)')
	plt.ylabel('PSD($V^2/Hz$)')
	plt.yscale('log')
	plt.grid()

def BPSKSignal(t,data,fSubPulse,tau,PRI,Np=1,delay=0):
	y   = np.zeros(len(t),dtype=np.complex64)
	ind_PRI = np.where((t >= delay) & (t <= PRI))[0]
	ind_tau = np.where((delay <= t) & (t <= delay+tau))[0]
	t = np.extract((delay<=t) & (t<=delay+tau),t)

	DutyCycle = tau/PRI
	print(DutyCycle)
	if DutyCycle*100 > 50:
		N = 1
	else:
		N = 2

	SamplesPulse = round(len(t)*DutyCycle)
	SamplesSubPulse = int(SamplesPulse/len(data))*N
	print(len(t),SamplesPulse,len(data),SamplesSubPulse)
	
	TSubPulse = tau/len(data)
	fSubPulse = 1/TSubPulse

	PhaseSubPulse = np.repeat(data,SamplesSubPulse)
	print(len(PhaseSubPulse),len(t),(len(t)-len(PhaseSubPulse)))
	PhaseSignal = np.append(PhaseSubPulse,np.zeros(np.abs(round(len(t)-len(PhaseSubPulse)))))

	for i in range(Np):
		y[ind_tau[0]+ind_PRI[-1]*i:ind_tau[-1]+1+ind_PRI[-1]*i] = np.sin(2*np.pi*fSubPulse*(t-delay)+PhaseSignal*np.pi)

	return y

# ------------------------- Generación del Chirp Periódico: ---------------------------------
# Parámetros del chirp, SI units
PRF        		= 1000                         # Hz, frecuencia de repetición de pulso,
PRI        		= 1/PRF                       # s, intervalo de repetición de pulso
DutyCycle   	= 50                         # %, ciclo de trabajo
tau        		= DutyCycle*PRI/100           # s, Ancho de pulso
fi          	= -15e6                      # Hz, frecuencia central/inicial
fo          	= +15e6                      # Hz, Frecuencia final
B            	= (fo-fi)                     # Hz, Ancho de banda del crirp
Np          	= 1                               # Nº, Número de pulso
R_virtual   	= 0                              # m, Rango del objetivo simulado
c            	= 299792458.0                 # m/s, velocidad de la luz
delay        	= 2*R_virtual/c             # s, delay de asociado a un target a distancia R_virtual
Noise_onoff 	= 0                               # 0 = off, 1 = on, Ruido añadido a la señal
Ap           	= 0.4                         # W, potencia de la señal
ti          	= 0                               # s, tiempo inicial de simulación
tf          	= delay+Np*PRI                # s, tiempo final de simulación
fs          	= 5*B                       # muestras/s, frecuencia de sampleo
N            	= int(fs*tf)                    # muestras, Número de muestras en total
t            	= np.linspace(ti,tf,N)       # s, vector tiempo

# ------------------------- Procesamiento de datos ---------------------------------

Noise = GaussNoiseSignal(t,0.02,0.02)

y_chirp = Ap*ChirpSignal(t,tau,PRI,fi,B,Np,delay) + Noise_onoff * Noise
TBP_chirp = tau*B

data = np.array([0, 1, 0, 1, 0, 1, 0, 1])
Tchip = tau/len(data)
fSubPulse = 1/Tchip
y_BPSK = Ap*BPSKSignal(t,data,fSubPulse,tau,PRI,Np,delay) + Noise_onoff * Noise
TBP_BPSK = tau/Tchip

print(f"\nChirp Signal Parameters:")
print(f"PRF        	= {PRF} Hz")
print(f"PRI        	= {PRI*1000} ms")
print(f"DutyCycle	= {round(DutyCycle)} %")
print(f"fi        	= {round(fi)*1e-6} MHz")
print(f"fo        	= {round(fo)*1e-6} MHz")
print(f"B        	= {round(B)*1e-6} MHz")
print(f"TBP        	= {round(TBP_chirp)}")

print(f"\nBPSK Signal Parameters:")
print(f"PRF        	= {PRF} Hz")
print(f"PRI        	= {PRI*1000} ms")
print(f"Bits Number = {len(data)} bits")
print(f"DutyCycle	= {round(DutyCycle)} %")
print(f"fSubPulse 	= {round(fSubPulse)*1e-3} kHz")
print(f"Tchip 		= {round(Tchip*1e6)} us")
print(f"TBP        	= {round(TBP_BPSK)}")

SignalAnalysis(t,y_chirp,fs,"Chirp")
SignalAnalysis(t,y_BPSK,fs,"BPSK")

FilteredChirpSignal = np.correlate(y_chirp, y_chirp, mode='same')
FilteredBPSKSignal = np.correlate(y_BPSK, y_BPSK, mode='same')

SignalAnalysis(t,FilteredChirpSignal,fs,"Filtered Chirp correlate")
SignalAnalysis(t,FilteredBPSKSignal,fs,"Filtered BPSK correlate")

print("correlate")
print(f"phro xy chirp {np.corrcoef(y_chirp, y_chirp)[0, 1]}")
print(f"phro xy BPSK {np.corrcoef(y_BPSK, y_BPSK)[0, 1]}")



h_Chirp = np.conj(y_chirp[::-1])
h_BPSK = np.conj(y_BPSK[::-1])

FilteredChirpSignal = np.convolve(y_chirp, h_Chirp, mode='same')
FilteredBPSKSignal = np.convolve(y_BPSK, h_BPSK, mode='same')




SignalAnalysis(t,FilteredChirpSignal,fs,"Filtered Chirp convolve")
SignalAnalysis(t,FilteredBPSKSignal,fs,"Filtered BPSK convolve")


print("Convolve")
print(f"phro xy chirp {np.corrcoef(y_chirp, h_Chirp)[0, 1]}")
print(f"phro xy BPSK {np.corrcoef(y_BPSK, h_BPSK)[0, 1]}")

plt.show()

exit()
plt.figure("1")
plt.subplot(221)
plt.title("Correlate")
plt.plot(FilteredChirpSignal)
plt.subplot(222)
plt.plot(np.abs(np.fft.fft(FilteredChirpSignal)))
plt.subplot(223)
plt.plot(FilteredBPSKSignal)
plt.subplot(224)
plt.plot(np.abs(np.fft.fft(FilteredBPSKSignal)))

plt.figure("2 Convolve")

plt.subplot(221)
plt.title("Convolve")
plt.plot(FilteredChirpSignal)
plt.subplot(222)
plt.plot(np.abs(np.fft.fft(FilteredChirpSignal)))
plt.subplot(223)
plt.plot(FilteredBPSKSignal)
plt.subplot(224)
plt.plot(np.abs(np.fft.fft(FilteredBPSKSignal)))


