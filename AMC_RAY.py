import numpy as np
import commpy.modulation as modulation
import matplotlib.pyplot as plt
import commpy.utilities as util
import seaborn as sns
from numpy import isrealobj
from numpy.random import standard_normal
from viterbi import Viterbi
from digcommpy.modulators import QamModulator, BpskModulator
from digcommpy.demodulators import QamDemodulator, BpskDemodulator
from pyphysim.modulators import ofdm
from pyphysim.modulators import QAM, QPSK, PSK
import math
import pandas as pd
from tqdm import tqdm
import time
from scipy import signal
from numpy import sqrt
from sk_dsp_comm import digitalcom as dc
from channel import channel_model

start_time = time.time()

def BER_calc(a, b):
    num_errors = np.sum(a != b)
    bit_error_rate = num_errors / len(a)
    return bit_error_rate

def awgn(s, SNRdB, L=1):
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if isrealobj(s):  # check if input is real/complex object type
        n = math.sqrt(N0 / 2) * standard_normal(s.shape)  # computed noise
    else:
        n = math.sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
    r = s + n  # received signal
    return r

def rayleigh(N):
    # N is the number of taps within the channel impulse response
    # Factor for exponentially decaying power profile:
    c_att = 100
    # Calculate variances of channel taps according to exponential decaying power profile
    var_ch = np.exp(-np.arange(N) / c_att)
    var_ch = var_ch / np.sum(var_ch)  # Normalize overall average channel power to one
    # Generate random channel coefficients (Rayleigh fading)
    h = np.sqrt(0.5) * (np.random.randn(N) + 1j * np.random.randn(N))* np.sqrt(var_ch)  # Complex Gaussian random numbers
    return h

# Define puncturing patterns
puncture_matrix_3_4 = np.array([[1, 1, 0], [1, 0, 1]])
puncturing_pattern_3_4 = [1,1,1,0,0,1]
puncture_matrix_5_6 = np.array([[1, 1, 0, 1, 1], [1, 0, 1, 0,0]])
puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
num_ones = puncturing_pattern_5_6.count(1)
numbers = [6, 6, 4, 10, 184]
lcm_value = np.lcm.reduce(numbers)
num_chunks = 3 # Number of chunks
N = lcm_value # or len(puncturing_pattern_5_6)

# Data generation
data_bits = np.random.randint(0, 2, N * num_chunks)  # Generate data for the entire transmission
print('data_bits=',len(data_bits))

# System parameters
fft_size = 128
Npc = 16
Npb = 10
num_used_subcarriers = 128 - Npc
num_ofdm_symbols = 100
num_symbols = num_ofdm_symbols * num_used_subcarriers # Number of QAM symbols that will be generated
cp_size = 32  # Size of the OFDM cyclic interval (in samples)
N_taps = 4

# Channel model (rayleigh and awgn) 
rayleighchan = rayleigh(N_taps) #initialize random rayleigh channel

def run_monte_carlo_simulation(snr_values_dB, modem, num_trials, puncturing_pattern, rate, K):
    avg_ber_values_coded = []
    all_H_est = []
    N_frame = 6
    fft_size = 92+8
    fft_size=64
    num_used_subcarriers=64
    cp_size=10
    Ofdm = ofdm.OFDM(fft_size, cp_size, num_used_subcarriers)
    for snr in tqdm(snr_values_dB):
        ber_coded = []
        snr_adjusted = snr + 10 * np.log10(K*rate*num_used_subcarriers/fft_size)# adjust SNR by code rate
        for _ in range(num_trials):
            errbits_coded = 0
            if puncturing_pattern is not None:
                v = Viterbi(7, [0o133, 0o171], puncturing_pattern)
                coded_bits = v.encode(data_bits)
                modulated = modem.modulate(coded_bits)  # Modulation
                after_ifft = Ofdm.modulate(modulated)
                Tx_out = np.fft.fft(after_ifft, n=len(after_ifft)) #the signal in frequency domain after TX
                #Channel distortion: rayleigh fading and AWGN
                # Calculate corresponding frequency response (needed for receiver part)
                h_zp = np.concatenate((rayleighchan, np.zeros(len(after_ifft) - N_taps)))  # Zero-padded channel impulse response (length Nc)
                H_exact = np.fft.fft(h_zp)  # Corresponding FFT
                # Convolution with channel impulse response 
                c_out = signal.lfilter(h_zp, 1, after_ifft)  # Apply channel distortion
                #Channel: 
                noisy_coded = awgn(c_out, snr_adjusted)  # Es/N0 = 100 dB
                Rx_in = np.fft.fft(noisy_coded, n=len(noisy_coded)) #the signal in frequency domain after channel distortions
                # LS Channel Estimation
                pilot_indices = np.arange(0, len(Tx_out), 4) #Assuming we have pilot positions and values (example with every 4th subcarrier as a pilot)
                X_pilots = Tx_out[pilot_indices]
                Y_pilots = Rx_in[pilot_indices]
                H_LS = Y_pilots / X_pilots
                # Interpolate LS estimates to all subcarriers (linear interpolation)
                H_est = np.interp(np.arange(len(after_ifft)), pilot_indices, H_LS)
                # Equalization
                G_ZF = 1.0 / H_exact  # Zero-forcing equalizer
                Rx_fft = np.fft.fft(noisy_coded)
                equ_op_fft = Rx_fft/H_exact
                equ_op = np.fft.ifft(equ_op_fft)  # inverse FFT to obtain time-domain equalized signal
                rx = Ofdm.demodulate(equ_op)
                demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                decoded_hard = v.decode(demodulated)
                errbits_coded += util.hamming_dist(data_bits,
                                                   decoded_hard[:len(data_bits)])  # Count the number of bit errors
            else:
                v = Viterbi(7, [0o133, 0o171])
                coded_bits = v.encode(data_bits)
                modulated = modem.modulate(coded_bits)  # Modulation
                after_ifft = Ofdm.modulate(modulated) #modulated ofdm signal in time 
                Tx_out = np.fft.fft(after_ifft, n=len(after_ifft)) #the signal in frequency domain after TX
                # Assuming we have pilot positions and values (example with every 4th subcarrier as a pilot)
                pilot_indices = np.arange(0, len(Tx_out), 4)
                #Channel distortion: rayleigh fading and AWGN
                # Calculate corresponding frequency response (needed for receiver part)
                h_zp = np.concatenate((rayleighchan, np.zeros(len(after_ifft) - N_taps)))  # Zero-padded channel impulse response (length Nc)
                H_exact = np.fft.fft(h_zp)  # Corresponding FFT
                # Convolution with channel impulse response 
                c_out = signal.lfilter(h_zp, 1, after_ifft)  # Apply channel distortion
                #Channel: 
                noisy_coded = awgn(c_out, snr_adjusted)  # Es/N0 = 100 dB
                Rx_in = np.fft.fft(noisy_coded, n=len(noisy_coded)) #the signal in frequency domain after channel distortions
                # LS Channel Estimation
                X_pilots = Tx_out[pilot_indices]
                Y_pilots = Rx_in[pilot_indices]
                H_LS = Y_pilots / X_pilots
                # Interpolate LS estimates to all subcarriers (linear interpolation)
                H_est = np.interp(np.arange(len(after_ifft)), pilot_indices, H_LS)
                # Equalization
                G_ZF = 1.0 / H_exact  # Zero-forcing equalizer
                Rx_fft = np.fft.fft(noisy_coded)
                equ_op_fft = Rx_fft/H_exact
                equ_op = np.fft.ifft(equ_op_fft)  # inverse FFT to obtain time-domain equalized signal
                rx = Ofdm.demodulate(equ_op)
                demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                decoded_hard = v.decode(demodulated)
                errbits_coded += util.hamming_dist(data_bits,
                                                   decoded_hard[:len(data_bits)])  # Count the number of bit errors
            ber_coded.append(errbits_coded / len(data_bits))
            all_H_est.append(H_est)

        ber_trials_coded_avg = np.mean(ber_coded)

        avg_ber_values_coded.append(ber_trials_coded_avg)
    #Plotting the true channel vs. the estimated channel
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(np.abs(H_exact))), np.abs(H_exact))
    #plt.plot(np.arange(len(np.abs(H_est))), np.abs(H_est), label='Estimated Channel (LS)', linestyle='--')
    plt.title('Rayleigh flat fading')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Magnitude')
    #plt.legend()
    plt.grid(True)
    plt.show()
    return avg_ber_values_coded, all_H_est

step_size = 0.5
snr_values_dB21= np.arange(0, 20, step_size)
snr_values_dB22= np.arange(0, 20, step_size)
snr_values_dB23= np.arange(0, 25, step_size)
snr_values_dB24= np.arange(0, 30, step_size)
snr_values_dB25= np.arange(0, 30, step_size)

snr_adjusted1 = snr_values_dB21 + 10 * np.log10(1)  # adjust SNR by code rate
snr_adjusted2 = snr_values_dB22  + 10 * np.log10(2)  # adjust SNR by code rate
snr_adjusted3 = snr_values_dB23  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted4 = snr_values_dB24  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted5 = snr_values_dB25  + 10 * np.log10(6)   # adjust SNR by code rate

# Run Monte Carlo simulations
y_BPSK_12, CSI_BPSK_12 = run_monte_carlo_simulation(snr_values_dB21, modulation.PSKModem(2), 30, puncturing_pattern=None, rate=1/2, K=1)
y_4QAM_34, CSI_4QAM_34 = run_monte_carlo_simulation(snr_values_dB22, modulation.QAMModem(4), 30, puncturing_pattern=puncturing_pattern_3_4, rate=3/4, K=2)
y_16QAM_12, CSI_16QAM_12 = run_monte_carlo_simulation(snr_values_dB23, modulation.QAMModem(16), 30, puncturing_pattern=None, rate=1/2, K=4)
y_16QAM_34, CSI_16QAM_34 = run_monte_carlo_simulation(snr_values_dB24, modulation.QAMModem(16), 30, puncturing_pattern=puncturing_pattern_3_4, rate=3/4, K=4)
y_64QAM_34, CSI_64QAM_34 = run_monte_carlo_simulation(snr_values_dB25, modulation.QAMModem(64), 30, puncturing_pattern=puncturing_pattern_3_4, rate=3/4, K=6)

# Plot the results
plt.semilogy(snr_adjusted1, y_BPSK_12, label='BPSK, r=1/2')
plt.semilogy(snr_adjusted2, y_4QAM_34, label='4QAM, r=3/4')
plt.semilogy(snr_adjusted3, y_16QAM_12, label='16QAM, r=1/2')
plt.semilogy(snr_adjusted4, y_16QAM_34, label='16QAM, r=3/4')
plt.semilogy(snr_adjusted5, y_64QAM_34, label='64QAM, r=3/4')
plt.title('BER vs. SNR')
plt.xlabel('SNR/Eb0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True)
plt.show()

#for data generation
rates = [1/2, 3/4]

# Ensure CSI has the same length as BER for each modulation scheme
def ensure_csi_length(csi_list, ber_list):
    csi_avg_list = []
    for i in range(len(ber_list)):
        csi_avg_list.append(np.mean(np.abs(csi_list[i * 30:(i + 1) * 30]), axis=0))
    return csi_avg_list

CSI_BPSK_12 = ensure_csi_length(CSI_BPSK_12, y_BPSK_12)
CSI_4QAM_34 = ensure_csi_length(CSI_4QAM_34, y_4QAM_34)
CSI_16QAM_12 = ensure_csi_length(CSI_16QAM_12, y_16QAM_12)
CSI_16QAM_34 = ensure_csi_length(CSI_16QAM_34, y_16QAM_34)
CSI_64QAM_34 = ensure_csi_length(CSI_64QAM_34, y_64QAM_34)

# Create DataFrames for adaptive modulation results
df_bpsk_12 = pd.DataFrame({'SNR (dB)': snr_adjusted1, 'Rate': rates[0], 'modulation': 'bpsk', 'BER': y_BPSK_12, 'CSI': CSI_BPSK_12})
df_4qam_34 = pd.DataFrame({'SNR (dB)': snr_adjusted2, 'Rate': rates[1], 'modulation': '4qam', 'BER': y_4QAM_34, 'CSI': CSI_4QAM_34})
df_qam16_12 = pd.DataFrame({'SNR (dB)': snr_adjusted3, 'Rate': rates[0], 'modulation': 'qam16', 'BER': y_16QAM_12, 'CSI': CSI_16QAM_12})
df_qam16_34 = pd.DataFrame({'SNR (dB)': snr_adjusted4, 'Rate': rates[1], 'modulation': 'qam16', 'BER': y_16QAM_34, 'CSI': CSI_16QAM_34})
df_qam64_34 = pd.DataFrame({'SNR (dB)': snr_adjusted5, 'Rate': rates[1], 'modulation': 'qam64', 'BER': y_64QAM_34, 'CSI': CSI_64QAM_34})

# Initialize an empty DataFrame to store the results
result_data = pd.DataFrame(columns=['SNR (dB)', 'Rate', 'modulation', 'BER', 'CSI'])

# Concatenate the DataFrames
concatenated_df = pd.concat([df_bpsk_12, df_4qam_34, df_qam16_12, df_qam16_34, df_qam64_34])

#filtering for threshold
threshold = 1e-1 #BER tolerable up to 10%
df_bpsk_12 = df_bpsk_12[df_bpsk_12['BER'] <= threshold]
df_4qam_34 = df_4qam_34[df_4qam_34['BER'] <= threshold]
df_qam16_12 = df_qam16_12[df_qam16_12['BER'] <= threshold]
df_qam16_34 = df_qam16_34[df_qam16_34['BER'] <= threshold]
df_qam64_34 = df_qam64_34[df_qam64_34['BER'] <= threshold]

#Getting snr thresholds for each scheme
SNR_threshold_bpsk_12 = df_bpsk_12['SNR (dB)'].min()
SNR_threshold_4qam_34 = df_4qam_34['SNR (dB)'].min()
SNR_threshold_qam16_12 = df_qam16_12 ['SNR (dB)'].min()
SNR_threshold_qam16_34 = df_qam16_34['SNR (dB)'].min()
SNR_threshold_qam64_34 = df_qam64_34['SNR (dB)'].min()

#Applying the snr threshold to filter 
df_qam16_34 = df_qam16_34[df_qam16_34 ['SNR (dB)'] < SNR_threshold_qam64_34]
df_qam16_12 = df_qam16_12[df_qam16_12['SNR (dB)'] < SNR_threshold_qam16_34]
df_4qam_34 = df_4qam_34[df_4qam_34 ['SNR (dB)'] < SNR_threshold_qam16_12]
df_bpsk_12 = df_bpsk_12[df_bpsk_12['SNR (dB)'] < SNR_threshold_4qam_34]

# Initialize an empty DataFrame to store the results
result_data = pd.DataFrame(columns=['SNR (dB)', 'Rate', 'modulation', 'BER', 'CSI'])

# Concatenate the DataFrames
result_data = pd.concat([df_bpsk_12, df_4qam_34, df_qam16_12, df_qam16_34, df_qam64_34])

# Define modulation orders for different modulations
modulation_orders = {'bpsk': 2, '4qam': 4, 'qam16': 16, 'qam64': 64}  # Add more if needed

# Add modulation order column
result_data['modulation_order'] = result_data['modulation'].map(modulation_orders)

# Combine modulation and rate into one column
result_data['modulation_rate'] = result_data['modulation'] + '_' + result_data['Rate'].astype(str)

# Calculate the average CSI for each row and replace the array with the average value
result_data['CSI'] = result_data['CSI'].apply(lambda x: np.mean(x))

# Use factorize() to assign unique indices to each unique value in the 'modulation_rate' column
unique_indices, unique_labels = pd.factorize(result_data['modulation_rate'])

# Subtract the minimum index value to start indices from zero
unique_indices -= unique_indices.min()

# Add the unique indices as a new column named 'modulation_rate_index'
result_data['modulation_rate_index'] = unique_indices

# Sort by modulation order and rate to prioritize higher rates
result_data.sort_values(by=['modulation_rate_index'], ascending=[False], inplace=True)

# Drop the intermediate columns if they are not needed
result_data.drop(['modulation_order','Rate','modulation'], axis=1, inplace=True)

# Specify the file path where you want to save the CSV file
file_path = "Data_RAY_4taps.csv"

# Save the DataFrame to a CSV file at the specified location
result_data .to_csv(file_path, index=False)

print("DataFrame saved to CSV file:", file_path)
print(result_data )

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time/60} minutes")

