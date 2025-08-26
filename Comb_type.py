# xxxxxxxxxxxx Importing Necessary Packages xxxxxxxxxxxx
import matplotlib.pyplot as plt
import numpy as np
import channels
import scipy.interpolate

from scipy import signal
from sk_dsp_comm import digitalcom as dc


np.random.seed(55)
# xxxxxxxxxxxxx Parameters xxxxxxxxxxxxxxx
M = 4  # Size of the modulation constelation
fft_size = 128
Npc = 16
Npb = 10
num_used_subcarriers = 128 - Npc
num_ofdm_symbols = 100
num_symbols = num_ofdm_symbols * num_used_subcarriers # Number of QAM symbols that will be generated
cp_size = 32  # Size of the OFDM cyclic interval (in samples)
pilot_value = 1+1j
pilot_type = 'comb'
ch_est_tech = 'MMSE'
beta = 17/9
L = 3
use_DFT = 0
SNR = 20 # in dB


# xxxxxxxxxxxxxx Transmitter xxxxxxxxxxxxxxxx
if pilot_type == 'comb':
    modulated_symbols, b, tx_data = dc.qam_gray_encode_bb(num_symbols, 1, M)
    Sp = int(fft_size/Npc)
    St = 0
    all_carriers = np.arange(fft_size)
    p_array = all_carriers[::Sp]
    d_array = np.delete(all_carriers, p_array)
    symbol = np.zeros((num_ofdm_symbols, fft_size), dtype='complex')
    symbol[:, p_array] = pilot_value
    symbol[:, d_array] = modulated_symbols.reshape(-1, num_used_subcarriers)
    symbolp2s = symbol.flatten()
    TX_out = dc.ofdm_tx(symbolp2s, fft_size, fft_size, 0, True, cp_size)
elif pilot_type == 'block':
    modulated_symbols, b, tx_data = dc.qam_gray_encode_bb(num_symbols, 1, M)
    symbol = dc.mux_pilot_blocks(modulated_symbols.reshape(-1, num_used_subcarriers), Npb)
    all_carriers = np.arange(fft_size)
    pilots = np.ones(fft_size) * pilot_value
    p_loc = np.arange(symbol.shape[0])[::Npb]
    d_loc = np.delete(np.arange(symbol.shape[0]), p_loc)
    symbol[p_loc, :] = pilots
    symbol = symbol.flatten()
    TX_out = dc.ofdm_tx(symbol, num_used_subcarriers, fft_size, 0, True, cp_size)
elif pilot_type == 'recursive':
    modulated_symbols, b, tx_data = dc.qam_gray_encode_bb(num_symbols, 1, M)
    TX_out = dc.ofdm_tx(modulated_symbols, num_used_subcarriers, fft_size, Npb, True, cp_size)

# extract real part
x = [ele.real for ele in modulated_symbols]
# extract imaginary part
y = [ele.imag for ele in modulated_symbols]

# plot the complex numbers
plt.scatter(x, y)
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.grid()
plt.show()

# xxxxxxxxxxxxxxx Channel Model xxxxxxxxxxxxxxxxxxx

hc = channels.rayleighFading(L) # impulse response spanning five symbols
H_exact = np.fft.fft(hc, fft_size)
c_out = signal.lfilter(hc, 1, TX_out)  # Apply channel distortion
r_out = channels.awgn(c_out, SNR)  # Es/N0 = 100 dB

# xxxxxxxxxxxxxxx Receiver xxxxxxxxxxxxxxx
# ----- OFDM Demodulation -------
if pilot_type == 'recursive':
    z_out, H = dc.ofdm_rx(r_out, num_used_subcarriers, fft_size, Npb, True, cp_size,alpha=0.9 , ht=hc);
else:
    z_out, H = dc.ofdm_rx(r_out, fft_size, fft_size, 0, True, cp_size);

# ----- Channel Estimation ------

def DFT_based(h_hat,fft_size):
    H_hat_time = np.fft.ifft(h_hat, fft_size)
    H_hat_time_dft = np.zeros_like(H_hat_time)
    H_hat_time_dft[:L] = H_hat_time[:L]
    H_hat_freq = np.fft.fft(H_hat_time_dft, fft_size)
    return H_hat_freq
def ls_Ch_est (OFDM_demod,p_array):
    pilots = OFDM_demod[p_array]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilot_value  # divide by the transmitted pilot values

    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase
    # separately
    Hest_abs = scipy.interpolate.interp1d(p_array, abs(Hest_at_pilots), fill_value='extrapolate', kind='linear')(all_carriers)
    Hest_phase = scipy.interpolate.interp1d(p_array, np.angle(Hest_at_pilots), fill_value='extrapolate', kind='linear')(all_carriers)
    Hest = Hest_abs * np.exp(1j * Hest_phase)
    plt.plot(all_carriers, abs(Hest), label='Estimated channel via interpolation')
    if use_DFT == 1 and ch_est_tech != "MMSE":
        Hest = DFT_based(Hest,fft_size)

    plt.plot(all_carriers, abs(np.fft.fft(hc, fft_size)), label='Correct Channel')
    plt.stem(p_array, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(all_carriers, abs(Hest), label='Estimated channel via DFT')
    plt.grid(True);
    plt.xlabel('Carrier index');
    plt.ylabel('$|H(f)|$');
    plt.legend(fontsize=10)
    plt.ylim(0, 3)
    plt.show()
    return Hest

def MMSE_channelEstimate(SNRdb,beta,fft_size,H_tilde):
    # import pdb;pdb.set_trace()
    C_response = np.array(H_exact).reshape(-1, 1)
    C_response_H = np.conj(C_response).T
    R_HH = np.matmul(C_response, C_response_H)
    snr = 10 ** (SNRdb / 10)

    H_tilde = H_tilde.reshape(-1, 1)
    W = np.matmul(R_HH, np.linalg.inv(R_HH + (beta / snr) * np.eye(fft_size)))
    HhatLMMSE = np.matmul(W, H_tilde)
    MMSE_ch = HhatLMMSE.squeeze()
    plt.plot(all_carriers, abs(MMSE_ch), label='Estimated channel via MMSE')
    if use_DFT == 1 :
        MMSE_ch = DFT_based(MMSE_ch,fft_size)

    plt.plot(all_carriers, abs(np.fft.fft(hc, fft_size)), label='Correct Channel')
    plt.plot(all_carriers, abs(MMSE_ch), label='Estimated channel via DFT')
    plt.grid(True);
    plt.title('MMSE channel estimation')
    plt.xlabel('Carrier index');
    plt.ylabel('$|H(f)|$');
    plt.legend(fontsize=10)
    plt.ylim(0, 3)
    plt.show()

    return MMSE_ch

if pilot_type == 'comb':
    if ch_est_tech != 'MMSE':
        ch = ls_Ch_est(z_out[0:fft_size],p_array)
    else:
        ch_ls = ls_Ch_est(z_out, p_array)
        ch = MMSE_channelEstimate(SNR,beta,fft_size,ch_ls)
    equalized_symbols = z_out / np.tile(ch, (num_ofdm_symbols))
    # -------- Data Extraction ---------
    data_only = equalized_symbols.reshape(-1, fft_size)[:, d_array]
    data = dc.qam_gray_decode(data_only.flatten(), M)
elif pilot_type == 'block':
    equalized_symbols = np.zeros((num_ofdm_symbols+12,fft_size),dtype='complex')
    zz = z_out.reshape(-1, fft_size)
    if ch_est_tech != 'MMSE':
        ch = ls_Ch_est(zz[0,:], all_carriers)
    else:
        ch_ls = ls_Ch_est(zz[0,:], all_carriers)
        ch = MMSE_channelEstimate(SNR, beta, fft_size, ch_ls)
    equalized_symbols = zz / np.tile(ch, (112,1))
    # -------- Data Extraction ---------
    data_only = equalized_symbols[d_loc, :]
    data = dc.qam_gray_decode(data_only.flatten(), M)
else:
    data = dc.qam_gray_decode(z_out, M)



plt.plot(equalized_symbols.real,equalized_symbols.imag,'.') # allow settling time
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.axis('equal')
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()

# xxxxxxxxxxx BER xxxxxxxxxxxxxxxxx

def calculate_bit_error_rate(original_data, received_data):
    num_errors = np.sum(original_data != received_data)
    bit_error_rate = num_errors / len(original_data)
    return bit_error_rate

ber = calculate_bit_error_rate(tx_data,data)
print(ber)
print(np.mean(np.abs(modulated_symbols)**2))