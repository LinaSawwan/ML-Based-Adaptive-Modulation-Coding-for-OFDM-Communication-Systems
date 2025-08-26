import channels
from scipy import signal
import numpy as np

# xxxxxxxxxxxxxxx Channel Model xxxxxxxxxxxxxxxxxxx
def channel(TX_out,SNR,L,fft_size):
    hc = channels.rayleighFading(L) # impulse response spanning five symbols
    H_exact = np.fft.fft(hc, fft_size)
    c_out = signal.lfilter(hc, 1, TX_out)  # Apply channel distortion
    r_out = channels.awgn(c_out, SNR)  # Es/N0 = 100 dB
    faded_signal = TX_out * hc
    received_signal = channels.awgn(faded_signal, SNR)
    equalized_r = received_signal/hc
    return r_out,equalized_r,H_exact