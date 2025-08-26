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
from pyphysim.modulators import QAM, QPSK, PSK
import math
import pandas as pd
from tqdm import tqdm
import time
from scipy import signal
from numpy import sqrt
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

def rayleighFading(N):
    """
    Generate Rayleigh flat-fading channel samples
    Parameters:
        N : number of samples to generate
    Returns:
        abs_h : Rayleigh flat fading samples
    """
    # 1 tap complex gaussian filter
    h = 1/sqrt(2)*(standard_normal(N)+1j*standard_normal(N))
    return abs(h)

# Define puncturing patterns
puncture_matrix_3_4 = np.array([[1, 1, 0], [1, 0, 1]])
puncturing_pattern_3_4 = [1,1,1,0,0,1]
puncture_matrix_5_6 = np.array([[1, 1, 0, 1, 1], [1, 0, 1, 0,0]])
puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
num_ones = puncturing_pattern_5_6.count(1)


numbers = [6, 6, 4, 10, 184]
lcm_value = np.lcm.reduce(numbers)

num_chunks = 5# Number of chunks
N = lcm_value # or len(puncturing_pattern_5_6)

def run_monte_carlo_simulation(snr_values_dB, modem, num_trials, puncturing_pattern, rate, K):
    avg_ber_values_coded = []
    avg_ber_values_uncoded = []
    N_frame = 6
    fft_size = 92+8
    num_used_subcarriers = 92
    cp_size = int(0.4*fft_size)
    Ofdm = ofdm.OFDM(fft_size, cp_size, num_used_subcarriers)
    Ray_channel = rayleighFading(3)
    for snr in tqdm(snr_values_dB):
        ber_coded = []
        ber_uncoded = []
        snr_adjusted = snr + 10 * np.log10(K*rate*num_used_subcarriers/fft_size)# adjust SNR by code rate
        for _ in range(num_trials):
            data_bits = np.random.randint(0, 2, N * num_chunks)  # Generate data for the entire transmission
            data_bits_chunks = np.array_split(data_bits, num_chunks)
            errbits_uncoded=0
            errbits_coded=0

            for chunk in data_bits_chunks:
                #print('chunk=',len(chunk), len(data_bits))
                if puncturing_pattern is not None:
                    v = Viterbi(7, [0o133, 0o171], puncturing_pattern)
                    coded_bits = v.encode(chunk)
                    #print('pun coded bits=', len(coded_bits))
                    modulated = modem.modulate(coded_bits)  # Modulation
                    #print('pun modulated=', len(modulated))
                    after_ifft = Ofdm.modulate(modulated)
                    #print('pun after_ifft=', after_ifft)
                    c_out = awgn(after_ifft, snr_adjusted)  # AWGN
                    noisy_coded = signal.lfilter(Ray_channel, 1, c_out)  # Apply channel distortion
                    rx = Ofdm.demodulate(noisy_coded)
                    demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                    #print(' pun demodulated',len(demodulated))
                    decoded_hard = v.decode(demodulated)
                    #print('pun decoded_hard', len(decoded_hard))

                else:
                    v = Viterbi(7, [0o133, 0o171])
                    coded_bits = v.encode(chunk)
                    modulated = modem.modulate(coded_bits)  # Modulation
                    after_ifft = Ofdm.modulate(modulated)
                    c_out = awgn(after_ifft, snr_adjusted)  # AWGN
                    noisy_coded = signal.lfilter(Ray_channel, 1, c_out)  # Apply channel distortion
                    rx = Ofdm.demodulate(noisy_coded)
                    demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                    decoded_hard = v.decode(demodulated)


                modulated_uncoded = modem.modulate(chunk)  # Modulation (uncoded case)
                noisy_uncoded = awgn(modulated_uncoded, snr)  # AWGN
                out_uncoded = signal.lfilter(Ray_channel, 1, noisy_uncoded)  # Apply channel distortion
                demodulated_uncoded=modem.demodulate(out_uncoded,demod_type='hard') # Demodulation (uncoded case)

                errbits_uncoded += util.hamming_dist(chunk, demodulated_uncoded[:len(chunk)])  # Count the number of bit errors
                errbits_coded += util.hamming_dist(chunk, decoded_hard[:len(chunk)])  # Count the number of bit errors

            ber_uncoded.append(errbits_uncoded/len(data_bits))
            ber_coded.append(errbits_coded/len(data_bits))

        ber_trials_coded_avg = np.mean(ber_coded)
        ber_trials_uncoded_avg = np.mean(ber_uncoded)

        avg_ber_values_coded.append(ber_trials_coded_avg)
        avg_ber_values_uncoded.append(ber_trials_uncoded_avg)


    return avg_ber_values_uncoded, avg_ber_values_coded

step_size = 5
snr_values_dB21= np.arange(0, 8, step_size)
snr_values_dB22= np.arange(0, 9, step_size)
snr_values_dB23= np.arange(0, 9, step_size)
snr_values_dB24= np.arange(0, 11, step_size)
snr_values_dB25= np.arange(0, 19, step_size)

snr_adjusted1 = snr_values_dB21 + 10 * np.log10(1)  # adjust SNR by code rate
snr_adjusted2 = snr_values_dB22  + 10 * np.log10(2)  # adjust SNR by code rate
snr_adjusted3 = snr_values_dB23  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted4 = snr_values_dB24  + 10 * np.log10(4)   # adjust SNR by code rate
snr_adjusted5 = snr_values_dB25  + 10 * np.log10(6)   # adjust SNR by code rate



x_BPSK_12, y_BPSK_12 = run_monte_carlo_simulation(snr_values_dB21 ,modulation.PSKModem(2), 25,puncturing_pattern=None,rate=1/2, K=1)
x_4QAM_34 ,y_4QAM_34 = run_monte_carlo_simulation(snr_values_dB22 ,modulation.QAMModem(4), 25, puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=2)
x_16QAM_12, y_16QAM_12= run_monte_carlo_simulation(snr_values_dB23 , modulation.QAMModem(16),25,puncturing_pattern=None,rate=1/2,K=4)
x_16QAM_34, y_16QAM_34 = run_monte_carlo_simulation(snr_values_dB24 , modulation.QAMModem(16),25,puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=4)
x_64QAM_34, y_64QAM_34 = run_monte_carlo_simulation(snr_values_dB25 , modulation.QAMModem(64),25,puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=6)




rates=[1/2,3/4]




# Create DataFrames for adaptive modulation results
df_bpsk_12 = pd.DataFrame({'SNR (dB)': snr_adjusted1,'Rate':rates[0],'modulation':'bpsk', 'BER': y_BPSK_12})
df_4qam_34 = pd.DataFrame({'SNR (dB)': snr_adjusted2,'Rate':rates[1],'modulation':'4qam', 'BER': y_4QAM_34})
df_qam16_12 = pd.DataFrame({'SNR (dB)': snr_adjusted3,'Rate':rates[0],'modulation':'qam16','BER': y_16QAM_12})
df_qam16_34 = pd.DataFrame({'SNR (dB)': snr_adjusted4,'Rate':rates[1],'modulation':'qam16','BER': y_16QAM_34})
df_qam64_34=pd.DataFrame({'SNR (dB)': snr_adjusted5,'Rate':rates[1],'modulation':'qam64','BER': y_64QAM_34})





# Initialize an empty DataFrame to store the results
result_data = pd.DataFrame(columns=['SNR (dB)', 'rate', 'modulation', 'BER'])

# Concatenate the DataFrames
concatenated_df = pd.concat([df_bpsk_12, df_4qam_34, df_qam16_12,df_qam16_34, df_qam64_34])

threshold = 1e-2
result_data = concatenated_df[concatenated_df['BER'] <= threshold]



# Extract minimum SNR and corresponding rate below threshold for each combination of modulation and rate
min_snr_rate_per_modulation_rate = result_data.groupby(['modulation', 'Rate']).apply(lambda x: x.loc[x['SNR (dB)'].idxmin()],group_keys=True)

# Reset index to make it a DataFrame
thresholds_df = min_snr_rate_per_modulation_rate.reset_index(drop=True)
result_data['modulation_rate'] = result_data['modulation'] + '_' + result_data['Rate'].astype(str)

# Drop the individual 'modulation' and 'Rate' columns if needed
result_data.drop(['modulation', 'Rate'], axis=1, inplace=True)

# Use factorize() to assign unique indices to each unique value in the 'modulation_rate' column
unique_indices, unique_labels = pd.factorize(result_data['modulation_rate'])

# Subtract the minimum index value to start indices from zero
unique_indices -= unique_indices.min()

# Add the unique indices as a new column named 'modulation_rate_index'
result_data['modulation_rate_index'] = unique_indices
result_data['SNR_linear'] = 10 ** (result_data['SNR (dB)'] / 10)

# Calculate capacity based on linear SNR
#result_data['capacity'] = np.log2(1 + result_data['SNR_linear'])

# Optionally, you might want to round the capacity to a certain number of decimal places
# result_data['capacity'] = result_data['capacity'].round(decimals=2)

# Drop the intermediate 'SNR_linear' column if it's not needed
result_data.drop('SNR_linear', axis=1, inplace=True)



# Specify the file path where you want to save the CSV file
file_path = "New_Data.csv"

# Save the DataFrame to a CSV file at the specified location
result_data.to_csv(file_path, index=False)

print("DataFrame saved to CSV file:",  file_path)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time/60} minutes")
