import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
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
    fft_signal_dBm = 10*np.log10(np.abs(fft_signal)**2 /0.001)
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

def GrapTFPSD(t,data,fs):

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
    fft_signal_dBm = 10*np.log10(np.abs(fft_signal)**2 /0.001)
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

def GaussNoiseSignal(t,mu,sigma):                         
    noise_signal = np.random.normal(mu, sigma, len(t)).astype(np.float32)*(1+1j)   
    return noise_signal

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

def RangeDet(delay):
    c = 299792458.0
    return c*delay/2

def SampleAdaptation(data_ref,t_ref,fs_ref,data_adap,t_adap):
    data_old = data_adap; t_old = t_adap
    data_adap = data_adap[0:len(data_ref)]
    t_adap = np.arange(0,len(data_ref)/fs_ref,1/fs_ref)

    print("\n ---- Adaptación de muestras ---- ")
    print("Long data ref        = {} muestras ".format(len(data_ref)))
    print("Long time ref        = {} muestras ".format(len(t_ref)))
    print("Long data old        = {} muestras ".format(len(data_old)))
    print("Long time old        = {} muestras ".format(len(t_old)))
    print("Long data adap       = {} muestras ".format(len(data_adap)))
    print("Long time adap       = {} muestras ".format(len(t_adap)))

    return t_adap, fs_ref, data_adap

def FunctionIntegration(t,f):
  fE = np.array([0])
  for i in range(len(t)):
    if i != range(len(t))[-1]:
      dt = t[i+1]-t[i]
      df = f[i]
      fE = np.append(fE,df*dt+fE[-1]) #CADA INDICE CORRESPONDE A LA
          # INTEGRAL DESDE TIEMPO t(0) A t(i)
  #E = fE[-1]  #INTEGRAL TOTAL
  E = np.amax(fE)
  return fE, E

def ReqParam(Rango):
    c = 299792458.0
    delay = 2*Rango/c
    fs_req = 1/delay
    bw_rangres_req = c/(2*Rango)
    return delay, fs_req, bw_rangres_req
