import matplotlib.pyplot as plt
from sk_dsp_comm import digitalcom as dc
from scipy import signal
import numpy as np
import channels

def calculate_bit_error_rate(original_data, received_data):
    num_errors = np.sum(original_data != received_data)
    bit_error_rate = num_errors / len(original_data)
    return bit_error_rate

def OFDM(SNR):
    hc = np.array([1.0])  # impulse response spanning five symbols
    # Quick example using the above channel with no cyclic prefix
    x1, b1, IQ_data1 = dc.qam_gray_encode_bb(520, 1, 2)
    x_out = dc.ofdm_tx(x1, 52, 64, 0, True, 0)
    c_out = signal.lfilter(hc, 1, x_out)  # Apply channel distortion
    r_out = channels.awgn(c_out, SNR)  # Es/N0 = 100 dB
    z_out, H = dc.ofdm_rx(r_out, 52, 64, -1, True, 0, alpha=0.95, ht=hc)
    bits = dc.qam_gray_decode(z_out, 2)
    ber = calculate_bit_error_rate(IQ_data1, bits)
    return ber,x_out

EbN0dB = np.arange(start=0,stop = 11,step = 1) # Eb/N0 range in dB for simulation
BER_thy = np.zeros(len(EbN0dB))
BER_sim = np.zeros(len(EbN0dB))
BER_temp = np.zeros(500)
for i,EbN0 in enumerate(EbN0dB):
    EsNo = EbN0 + 10 * np.log10((1*52) / 64)
    for j in range(500):
        BER_temp[j],ff = OFDM(EsNo)
    BER_sim[i] = np.mean(BER_temp)
    BER_thy[i] = dc.qam_bep_thy(EbN0,2,eb_n0_mode=False)
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,BER_sim,'r*',label='Simulation')
ax.semilogy(EbN0dB,BER_thy,'k-',label='Theoretical')
ax.set_title('Probability of Bit Error for BPSK over AWGN');
ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_ylabel(r'Probability of Bit Error - $P_b$');
ax.legend()
ax.grid(True,which = 'both')
ax.set_ylim([1e-5, 1])
fig.show()
