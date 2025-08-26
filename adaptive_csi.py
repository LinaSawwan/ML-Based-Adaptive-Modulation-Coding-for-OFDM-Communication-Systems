import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation
import matplotlib.pyplot as plt
import commpy.utilities as util
from numpy import isrealobj
from numpy.random import standard_normal
from viterbi import Viterbi
from digcommpy.modulators import QamModulator, BpskModulator
from digcommpy.demodulators import QamDemodulator, BpskDemodulator
from pyphysim.modulators import ofdm
from pyphysim.modulators import QAM, QPSK, PSK, BPSK
import math
import pandas as pd
from tqdm import tqdm
import time
from scipy import signal
from numpy import sqrt
from sk_dsp_comm import digitalcom as dc
from channel import channel_model
import pylab as pyl
plt.ion()
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

def rayleighFading(x):
    """
    Generate Rayleigh flat-fading channel samples with multiple taps

    Parameters:
    N : number of samples to generate
    num_taps : number of taps in the channel

    Returns:
    h : Rayleigh flat fading samples for each tap
    """
    N = x
    h_tap = []
    # Generate complex Gaussian random numbers for each tap
    h_tap = 1/sqrt(2)*(standard_normal(N)+1j*standard_normal(N))

    return h_tap

# Define puncturing patterns
puncture_matrix_3_4 = np.array([[1, 1, 0], [1, 0, 1]])
puncturing_pattern_3_4 = [1,1,1,0,0,1]
puncture_matrix_5_6 = np.array([[1, 1, 0, 1, 1], [1, 0, 1, 0,0]])
puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
num_ones = puncturing_pattern_5_6.count(1)


numbers = [6, 6, 4, 10, 184]
lcm_value = np.lcm.reduce(numbers)

num_chunks = 2# Number of chunks
N = lcm_value # or len(puncturing_pattern_5_6)


data_bits = np.random.randint(0, 2, N * num_chunks)  # Generate data for the entire transmission
print('data_bits=',len(data_bits))

#data_bits = np.random.randint(0, 2, 10000)  # Generate data for the entire transmission
def run_monte_carlo_simulation(snr_values_dB, modem, num_trials, puncturing_pattern, rate, K):
    avg_ber_values_coded = []
    N_frame = 6
    #fft_size = 92+8
    #num_used_subcarriers = 92
    fft_size=64
    num_used_subcarriers=52
    #cp_size = int(0.4*fft_size)
    cp_size=32
    Ofdm = ofdm.OFDM(fft_size, cp_size, num_used_subcarriers)
    #Ray_channel = rayleighFading(3)
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
                hc = rayleighFading(after_ifft)  # impulse response spanning five symbols
                #H_exact = np.fft.fft(hc, fft_size)
                c_out = hc * after_ifft # Apply channel distortion
                noisy_coded= awgn(c_out,snr_adjusted) # add noise to the transmitted signal
                received_signal = noisy_coded / hc
                #received_signal= noisy_coded/H_exact
                rx = Ofdm.demodulate(received_signal)
                demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                decoded_hard = v.decode(demodulated)
                errbits_coded += util.hamming_dist(data_bits,
                                                   decoded_hard[:len(data_bits)])  # Count the number of bit errors

            else:
                v = Viterbi(7, [0o133, 0o171])
                coded_bits = v.encode(data_bits)
                coded_bit = np.array(coded_bits)
                bpsk = BPSK()
                modulated = bpsk.modulate(coded_bit)
                #modulated = modem.modulate(coded_bits)  # Modulation
                after_ifft = Ofdm.modulate(modulated)
                hc = rayleighFading(1)  # impulse response spanning five symbols
                c_out = hc * after_ifft
                # c_out = signal.lfilter(hc, 1, after_ifft) 
                # Add noise
                #noisePower = 10 ** (-snr / 20)  # calculate the noise power for a given SNR value
                #noise = (noisePower) * 1 / np.sqrt(2) * (
                           # pyl.randn(len(after_ifft)) + 1j * pyl.randn(len(after_ifft)))  # generate noise
                noisy_coded= awgn(c_out,snr_adjusted)  # add noise to the transmitted signal
                #h_est = noisy_coded / after_ifft
                received_signal = noisy_coded / hc
                #received_signal = noisy_coded / H_exact
                rx = Ofdm.demodulate(received_signal)
                #demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                demodulated = bpsk.demodulate(rx)
                decoded_hard = v.decode(demodulated)
                errbits_coded += util.hamming_dist(data_bits,
                                                   decoded_hard[:len(data_bits)])  # Count the number of bit errors
            ber_coded.append(errbits_coded / len(data_bits))

        ber_trials_coded_avg = np.mean(ber_coded)
        avg_ber_values_coded.append(ber_trials_coded_avg)

    # Trying to extract the channel state information
    #Channel estimate h[k] = Y[k]/x[k] both in frequency domain
    print("coded bits")
    print(coded_bit)
    print(coded_bit.shape)
    print("Modulated is")
    print(modulated)
    print("Rx value is")
    print(rx)
    

    csi_fd = rx[:len(modulated)] / modulated
    CSI = np.fft.ifft(csi_fd, len(csi_fd))
    
    
    print("The CSI is")
    print(CSI)
    print(len(abs(CSI)))
    plt.figure(1)
    plt.plot(np.arange(len(CSI)), abs(np.fft.fft(hc, len(CSI))), label='Correct Channel')
    plt.stem(np.arange(len(CSI)), abs(CSI), label='Pilot estimates')
    plt.xlabel('Carrier index');
    plt.ylabel('$|H(f)|$');
    plt.legend(fontsize=10)
    plt.show()
    plt.pause(0.001)

        

    return  avg_ber_values_coded

step_size = 2
snr_values_dB21= np.arange(0, 30, step_size)
snr_values_dB22= np.arange(0, 30, step_size)
snr_values_dB23= np.arange(0, 30, step_size)
snr_values_dB24= np.arange(0, 30, step_size)
snr_values_dB25= np.arange(0, 30, step_size)

snr_adjusted1 = snr_values_dB21 + 10 * np.log10(1)  # adjust SNR by code rate
snr_adjusted2 = snr_values_dB22  + 10 * np.log10(2)  # adjust SNR by code rate
snr_adjusted3 = snr_values_dB23  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted4 = snr_values_dB24  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted5 = snr_values_dB25  + 10 * np.log10(6)   # adjust SNR by code rate



y_BPSK_12 = run_monte_carlo_simulation(snr_values_dB21 ,modulation.PSKModem(2), 30,puncturing_pattern=None,rate=1/2, K=1)
# y_4QAM_34 = run_monte_carlo_simulation(snr_values_dB22 ,modulation.QAMModem(4), 30, puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=2)
# y_16QAM_12= run_monte_carlo_simulation(snr_values_dB23 , modulation.QAMModem(16),30,puncturing_pattern=None,rate=1/2,K=4)
# y_16QAM_34 = run_monte_carlo_simulation(snr_values_dB24 , modulation.QAMModem(16),30,puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=4)
# y_64QAM_34 = run_monte_carlo_simulation(snr_values_dB25 , modulation.QAMModem(64),30,puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=6)




# Plot the results
plt.figure(2)
plt.semilogy(snr_values_dB21 , y_BPSK_12, label='BPSK, r=1/2')
# plt.semilogy(snr_values_dB22 ,y_4QAM_34, label='4QAM, r=3/4')
# plt.semilogy(snr_values_dB23 , y_16QAM_12, label='16QAM, r=1/2')
# plt.semilogy(snr_values_dB24 ,  y_16QAM_34, label='16QAM, r=3/4')
# plt.semilogy(snr_values_dB25 ,  y_64QAM_34, label='64QAM, r=3/4')

plt.title('BER vs. snr')
plt.xlabel('SNR/Eb0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True)



plt.show()
plt.pause(0.001) 
plt.show(block=True)




rates=[1/2,3/4]




# # Create DataFrames for adaptive modulation results
# df_bpsk_12 = pd.DataFrame({'SNR (dB)': snr_adjusted1,'Rate':rates[0],'modulation':'bpsk', 'BER': y_BPSK_12})
# # df_4qam_34 = pd.DataFrame({'SNR (dB)': snr_adjusted2,'Rate':rates[1],'modulation':'4qam', 'BER': y_4QAM_34})
# # df_qam16_12 = pd.DataFrame({'SNR (dB)': snr_adjusted3,'Rate':rates[0],'modulation':'qam16','BER': y_16QAM_12})
# # df_qam16_34 = pd.DataFrame({'SNR (dB)': snr_adjusted4,'Rate':rates[1],'modulation':'qam16','BER': y_16QAM_34})
# # df_qam64_34=pd.DataFrame({'SNR (dB)': snr_adjusted5,'Rate':rates[1],'modulation':'qam64','BER': y_64QAM_34})





# # Initialize an empty DataFrame to store the results
# result_data = pd.DataFrame(columns=['SNR (dB)', 'rate', 'modulation', 'BER'])

# # Concatenate the DataFrames
# concatenated_df = pd.concat([df_bpsk_12, df_4qam_34, df_qam16_12,df_qam16_34, df_qam64_34])

# threshold = 1e-2
# result_data = concatenated_df[concatenated_df['BER'] <= threshold]



# # Extract minimum SNR and corresponding rate below threshold for each combination of modulation and rate
# min_snr_rate_per_modulation_rate = result_data.groupby(['modulation', 'Rate']).apply(lambda x: x.loc[x['SNR (dB)'].idxmin()])

# # Reset index to make it a DataFrame
# thresholds_df = min_snr_rate_per_modulation_rate.reset_index(drop=True)
# result_data['modulation_rate'] = result_data['modulation'] + '_' + result_data['Rate'].astype(str)

# # Drop the individual 'modulation' and 'Rate' columns if needed
# #result_data.drop(['modulation', 'Rate'], axis=1, inplace=True)


# # Use factorize() to assign unique indices to each unique value in the 'modulation_rate' column
# unique_indices, unique_labels = pd.factorize(result_data['modulation_rate'])

# # Subtract the minimum index value to start indices from zero
# unique_indices -= unique_indices.min()

# # Add the unique indices as a new column named 'modulation_rate_index'
# result_data.loc[:, 'modulation_rate_index'] = unique_indices
# result_data.loc[:, 'SNR_linear'] = 10 ** (result_data['SNR (dB)'] / 10)

# # Calculate capacity based on linear SNR
# result_data.loc[:, 'capacity']  = np.log2(1 + result_data['SNR_linear'])
# # Define modulation orders for different modulations
# modulation_orders = {'bpsk': 2, '4qam': 4, 'qam16': 16, 'qam64': 64}  # Add more if needed

# # Calculate spectral efficiency
# #result_data['spectral_efficiency'] = np.log2(result_data['modulation'].map(modulation_orders)) * result_data['Rate']

# # Optionally, you might want to round the capacity to a certain number of decimal places
# #result_data.loc[:, 'capacity']  = result_data['capacity'].round(decimals=2)

# # Drop the individual 'modulation' and 'Rate' columns if needed
# result_data.drop(['modulation', 'Rate'], axis=1, inplace=True)

# # Drop the intermediate 'SNR_linear' column if it's not needed
# result_data.drop('SNR_linear', axis=1, inplace=True)



# # Specify the file path where you want to save the CSV file
# file_path = "DaraRay_0.5.csv"

# # Save the DataFNew_Datarame to a CSV file at the specified location
# result_data.to_csv(file_path, index=False)

# print("DataFrame saved to CSV file:",  file_path)
# print(result_data)


# end_time = time.time()
# execution_time = end_time - start_time

# print(f"Execution time: {execution_time/60} minutes")