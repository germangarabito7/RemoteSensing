#from FunctionsRadar import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal

def ChirpSignal(t,tau,PRI,fc,B,Np=1,delay=0):
    """ 
    La chirp(t,tao,PRI,fc,B,N=0):
    Genera un chirp periódico en un intervalo de tiempo t, ancho de pulso tau, 
    periodo PRI, frecuecia central fc, con crecimiento lineal en frecuencia para un 
    ancho de banda B y con N pulsos en intervalos de pulsos coherentes CPI = N*PRI. 
    N=0 por default te genera infinitos. 
    """
    y   = np.zeros(len(t),dtype=np.complex64)
    K   = B/tau
    ind_PRI = np.where((t >= delay) & (t <= PRI))[0]
    ind_tau = np.where((delay <= t) & (t <= delay+tau))[0]
    t = np.extract((delay<=t) & (t<=delay+tau),t)
    for i in range(Np):
        y[ind_tau[0]+ind_PRI[-1]*i:ind_tau[-1]+1+ind_PRI[-1]*i] = np.exp(1j*(2*np.pi*fc*(t-delay)+2*np.pi*K*((t-delay)**2)/2))

    return y

def SaveBinFile(data,FileName,SampleRate,ti,tf):
    np.array(data).tofile(FileName)
    print("\n ---- Guardado de archivo: {} ---- ".format(FileName))
    print("N        = {} muestras ".format(len(data)))
    print("fs       = {} MS/s ".format(SampleRate/1e6))
    print("it       = {} ns/muestra ".format(round(1/SampleRate*1e6,3)))
    print("ti       = {} s ".format(round(ti,3)))
    print("tf       = {} us ".format(round(tf*1e6,3)))
    return

def ReadBinFile(FileName,SampleRate):
    data = np.fromfile(FileName, dtype=np.complex64)
    t = np.linspace(0,len(data)/SampleRate,len(data), endpoint=True)       # s, vector tiempo

    print("\n ---- Lectura de archivo: {} ---- ".format(FileName))
    print("N        = {} muestras ".format(len(data)))
    print("fs       = {} MS/s ".format(SampleRate/1e6))
    print("it       = {} ns/muestra ".format(round(1/SampleRate*1e9,3)))
    print("tf       = {} s ".format(round(t[-1],3)))

    return t, SampleRate, data

def GrapTFPSD(t,data,N,fs):
    print(len(t),len(data))
    plt.subplot(311)
    plt.title("Señal")
    plt.plot(t,np.real(data),linewidth=0.5,label='Re(Data)')
    plt.plot(t,np.imag(data),linewidth=0.5,label='Img(Data)')
    plt.ylabel('Pt[W]')
    plt.xlabel('t[s]')
    plt.legend()
    plt.grid()

    #Plot de la FFT del chirp transmitido
    fft_signal = np.fft.fft(data); fft_signal = np.fft.fftshift(fft_signal)
    fft_signal_dBm = 10*np.log10((np.abs(fft_signal)/0.001))
    freqs = np.fft.fftfreq(len(data),1/fs); freqs = np.fft.fftshift(freqs)

    plt.subplot(312)
    plt.title('FFT')
    plt.plot(freqs,fft_signal_dBm,linewidth=0.5,label='Data')
    plt.ylabel('Pt[dBm]')
    plt.xlabel('f[Hz]')
    plt.legend()
    plt.grid()
    
    plt.subplot(313)
    plt.title("PSD")
    f, PSD_signal = signal.periodogram(np.real(data)+1j*np.imag(data),fs,window='boxcar',return_onesided=False,scaling='spectrum')
    plt.semilogy(f,PSD_signal,linewidth=0.5,label='Data')
    plt.xlabel('f[Hz]')
    plt.ylabel('PSD[V²/Hz]')
    plt.legend()
    plt.grid()
    return
# ------------------------- Generación del Chirp Periódico: ---------------------------------
# Parámetros del chirp, SI units
PRF         	= 1e3                     					# Hz, frecuencia de repetición de pulso,
PRI         	= 1/PRF                     				# s, intervalo de repetición de pulso
DutyCycle   	= 100                     					# %, ciclo de trabajo
tao         	= DutyCycle*PRI/100         				# s, Ancho de pulso
fi          	= -500e6                     					# Hz, frecuencia central/inicial
fo          	= +500e6                      				# Hz, Frecuencia final
B           	= (fo-fi)                   				# Hz, Ancho de banda del crirp
Np          	= 1                         				# Nº, Número de pulso
R_virtual_onoff	= 1                         				# 0 = off, 1 = on, Ruido añadido a la señal
R_virtual   	= 50                        				# m, Rango del objetivo simulado
c           	= 299792458.0               				# m/s, velocidad de la luz
delay       	= (2*R_virtual/c)*R_virtual_onoff			# s, delay de asociado a un target a distancia R_virtual
Noise_onoff 	= 0                         				# 0 = off, 1 = on, Ruido añadido a la señal
V 	        	= 0.2236*10                     					# W, potencia de la señal
ti          	= 0                         				# s, tiempo inicial de simulación
tf          	= delay+Np*PRI  							# s, tiempo final de simulación
fs          	= 2e9                   			# muestras/s, frecuencia de sampleo
N           	= int(fs*tf)                				# muestras, Número de muestras en total
t           	= np.linspace(ti,tf,N, endpoint=True)      	# s, vector tiempo

# ------------------------- Procesamiento de datos ---------------------------------
start_time = time.time()

signal_tx   = ((V)**2/50)*ChirpSignal(t,tao,PRI,fi,B,Np,delay) 
SaveBinFile(signal_tx,f"signal_tx_tao{tao}s_PRI{int(PRI/1e3)}kHz_fi{int(fi/1e6)}MHz_B{int(B/1e6)}MHz_delay{int(delay*1e6)}us.dat",fs,ti,tf)
GrapTFPSD(t,signal_tx,N,fs)
plt.show()
exit()
t_rx, fs_rx, signal = ReadBinFile("chirpsignal_5MHz_fs12MHz.dat",fs)
t_rx_etttus_eco, fs_rx_etttus_eco, signal_etttus_eco = ReadBinFile("signal_ettus_eco_rx.dat",fs)
t_rx_pluto_ref, fs_rx_pluto_ref, signal_pluto_ref = ReadBinFile("signal_pluto_ref_rx.dat",fs)

x = 50*N
i = len(signal) - x
j = len(signal_etttus_eco) - x
k = len(signal_pluto_ref) - x
s = -1

t_rx, fs_rx, signal = t_rx[i:s], fs_rx, signal[i:s]
t_rx_etttus_eco, fs_rx_etttus_eco, signal_etttus_eco = t_rx_etttus_eco[j:s], fs_rx_etttus_eco, signal_etttus_eco[j:s]
t_rx_pluto_ref, fs_rx_pluto_ref, signal_pluto_ref = t_rx_pluto_ref[k:s], fs_rx_pluto_ref, signal_pluto_ref[k:s]

#print("\nProcensando datos")
R = np.correlate(signal_pluto_ref, signal_etttus_eco, mode='full'); print("Correlación cruzada calculada")
t_corr = np.linspace(-t_rx_etttus_eco[-1], t_rx_etttus_eco[-1], len(R)) 
retardo = t_corr[np.argmax(np.abs(R))]; print("Delay de eco calculado")
rango = RangeDet(retardo)


plt.figure("Chirp")
GrapTFPSD(t_rx,signal,fs_rx)

plt.figure("Chirp Ref")
GrapTFPSD(t_rx_etttus_eco,signal_etttus_eco,fs_rx_etttus_eco)

plt.figure("Chirp Ec")
GrapTFPSD(t_rx_pluto_ref,signal_pluto_ref,fs_rx_pluto_ref)

plt.figure("Resultados de la correlación")
GrapTFPSD(t_corr*c/2,np.abs(np.abs(R)/np.amax(np.abs(R))),fs_rx_etttus_eco)

plt.show()
exit()

#t_tx, fs_tx, signal_tx = ReadBinFile("signal_trx.dat",fs)
#plt.figure("Chirp Tx")
#GrapTFPSD(t_tx,signal_tx,fs_tx)
#plt.show()


#noise       = GaussNoiseSignal(t,0,0.1)                                          					# Ruido gausiano
#signal_tx   = Pp*ChirpSignal(t,tao,PRI,fi,B,Np,0)                            						# Señal chirp transmitido
#signal_rx   = 0.8*Pp*ChirpSignal(t,tao,PRI,fi,B,Np,delay) + noise * Noise_onoff     # Señal chirp recibido

#SaveBinFile(signal_tx,"simsignal_tx.dat",fs,ti,tf)
#t_tx, fs_tx, signal_tx = ReadBinFile("simsignal_tx.dat",fs)
#t_tx, fs_tx, signal_tx = ReadBinFile("signal_trx.dat",fs)
#plt.figure("Chirp Tx")
#GrapTFPSD(t_tx,signal_tx,fs_tx)
#plt.show()

#SaveBinFile(signal_rx,"simsignal_rx.dat",fs,ti,tf)
#t_rx, fs_rx, signal_rx = ReadBinFile("simsignal_rx.dat",fs)
#t_rx, fs_rx, signal_rx = ReadBinFile("signal_rx2.dat",fs)

#FileName_tx = "ChirpSignal__PRF1.0KHz__DC11.9__B4.0MHz__fs6.0MHz__Noise0__Targ0m.dat"
#t_tx, fs_tx, signal_tx = ReadBinFile(FileName_tx,6e6)

#FileName_rx = "ChirpSignal__PRF1.0KHz__DC11.9__B4.0MHz__fs6.0MHz__Noise1__Targ10m.dat"
#t_rx, fs_rx, signal_rx = ReadBinFile(FileName_rx,6e6)

#print("\nProcensando datos")
#R = np.correlate(signal_rx, signal_tx, mode='full'); print("Correlación cruzada calculada")
#t_corr = np.linspace(-t_tx[-1], t_tx[-1], len(R)) 
#retardo = t_corr[np.argmax(np.abs(R))]; print("Delay de eco calculado")
#rango = RangeDet(retardo)

#end_time = time.time(); delta = end_time - start_time; 
#print(f"El tiempo de ejecución fue de {delta:.5f} s")


"""
# ------------------------- Resultados de simulación ---------------------------------
fr = 3
print("\n--------- Características requeridas de muestro dado el target -----------")
delay_tar, fs_req, bw_rangres_req = ReqParam(R_virtual)
print("Rango Virtual            = {} m".format(R_virtual))
print("delay                    = {} ns".format(round(delay_tar*1e9,fr)))
print("Frecuencia sampleo req.  = {} MS/s".format(round(fs_req/1e6,fr)))
print("Bandwidth req. 			= {} MHz".format(round(bw_rangres_req/1e6,fr)))

print('\n') 
print("              --- RESULTADOS: ---")
print("Frecuencia de sampleo    = {} MHz".format(round(fs*1e-6,fr)))
print("Delay generado           = {} ns".format(round(delay*1e9),fr))
print("Delay calculado          = {} ns".format(round(retardo*1e9,fr)))
print("Rango al objetivo        = {} m".format(round(rango,fr)))
print('Error Rango              = {} %:'.format(round(np.abs((rango-R_virtual)/R_virtual)*100,fr)))
"""
plt.figure("Chirp Tx")
GrapTFPSD(t_tx,signal_tx,fs_tx)
"""
plt.figure("Chirp Rx")
GrapTFPSD(t_rx,signal_rx,fs_rx)

plt.figure("Resultados de la correlación")
GrapTFPSD(t_corr,np.abs(np.abs(R)/np.amax(np.abs(R))),fs_rx)
"""
