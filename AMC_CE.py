import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation
import matplotlib.pyplot as plt
import commpy.utilities as util
from numpy import isrealobj
from numpy.random import standard_normal
from viterbi import Viterbi
from pyphysim.modulators import ofdm
from pyphysim.modulators import QAM, QPSK, PSK
import math

import pandas as pd
# Set random seed for reproducibility
np.random.seed(42)
def awgn(s, SNRdB, L=1):
    """
    AWGN channel

    Add AWGN noise to input signal. The function adds AWGN noise vector to signal
    's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power
    spectral density N0 of noise added

    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB)
            for the received signal
        L : oversampling factor (applicable for waveform simulation)
            default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
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


# Define puncturing patterns
puncture_matrix_3_4 = np.array([[1, 1, 0], [1, 0, 1]])
puncturing_pattern_3_4 = [1,1,1,0,0,1]
puncture_matrix_5_6 = np.array([[1, 1, 0, 1, 1], [1, 0, 1, 0,0]])
puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
num_ones = puncturing_pattern_5_6.count(1)
print(num_ones)

numbers = [6, 6, 4, 10, 184]
lcm_value = np.lcm.reduce(numbers)
print('lcm value=',lcm_value)
num_chunks = 5# Number of chunks
N = lcm_value # or len(puncturing_pattern_5_6)
qpsk_modulator = QPSK()  # QPSK
qam=QAM(4)
def run_monte_carlo_simulation(snr_values_dB, modem, num_trials, puncturing_pattern, rate, K):
    avg_ber_values_coded = []
    avg_ber_values_uncoded = []
    N_frame = 6

    fft_size = 92+8
    num_used_subcarriers = 92
    cp_size = int(0.4*fft_size)
    Ofdm = ofdm.OFDM(fft_size, cp_size, num_used_subcarriers)
    for snr in snr_values_dB:
        ber_coded = []
        ber_uncoded = []
        snr_adjusted = snr + 10 * np.log10(K*rate*num_used_subcarriers/fft_size)# adjust SNR by code rate
        for _ in range(num_trials):
            data_bits = np.random.randint(0, 2, N * num_chunks)  # Generate data for the entire transmission
            data_bits_chunks = np.array_split(data_bits, num_chunks)
            errbits_uncoded=0
            errbits_coded=0

            for chunk in data_bits_chunks:
                print('chunk=',len(chunk), len(data_bits))
                if puncturing_pattern is not None:
                    v = Viterbi(7, [0o133, 0o171], puncturing_pattern)
                    coded_bits = v.encode(chunk)
                    print('pun coded bits=', len(coded_bits))
                    modulated = modem.modulate(coded_bits)  # Modulation
                    print('pun modulated=', len(modulated))
                    after_ifft = Ofdm.modulate(modulated)
                    print('pun after_ifft=', len(after_ifft))
                    noisy_coded = awgn(after_ifft, snr_adjusted)  # AWGN
                    rx = Ofdm.demodulate(noisy_coded)
                    demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                    print(' pun demodulated',len(demodulated))
                    decoded_hard = v.decode(demodulated)
                    #print('pun decoded_hard', len(decoded_hard))

                else:
                    v = Viterbi(7, [0o133, 0o171])
                    coded_bits = v.encode(chunk)
                    modulated = modem.modulate(coded_bits)  # Modulation
                    after_ifft = Ofdm.modulate(modulated)
                    noisy_coded = awgn(after_ifft, snr_adjusted)  # AWGN
                    rx = Ofdm.demodulate(noisy_coded)
                    demodulated = modem.demodulate(rx,demod_type='hard')  # Demodulation (hard output)
                    decoded_hard = v.decode(demodulated)


                modulated_uncoded = modem.modulate(chunk)  # Modulation (uncoded case)
                noisy_uncoded = awgn(modulated_uncoded, snr)  # AWGN
                demodulated_uncoded=modem.demodulate(noisy_uncoded,demod_type='hard') # Demodulation (uncoded case)

                errbits_uncoded += util.hamming_dist(chunk, demodulated_uncoded[:len(chunk)])  # Count the number of bit errors
                errbits_coded += util.hamming_dist(chunk, decoded_hard[:len(chunk)])  # Count the number of bit errors

            ber_uncoded.append(errbits_uncoded/len(data_bits))
            ber_coded.append(errbits_coded/len(data_bits))

        ber_trials_coded_avg = np.mean(ber_coded)
        ber_trials_uncoded_avg = np.mean(ber_uncoded)

        avg_ber_values_coded.append(ber_trials_coded_avg)
        avg_ber_values_uncoded.append(ber_trials_uncoded_avg)

    print(f'Raw BER (Coded): {avg_ber_values_coded}')
    print(f'Raw BER (Uncoded): {avg_ber_values_uncoded}')
    return avg_ber_values_uncoded, avg_ber_values_coded


snr_values_dB21= np.arange(0, 26, 1)
snr_values_dB22= np.arange(0, 26, 1)
snr_values_dB23= np.arange(0, 26, 1)

snr_adjusted1 = snr_values_dB21 + 10 * np.log10(6)  # adjust SNR by code rate
snr_adjusted2 = snr_values_dB22  + 10 * np.log10(6)  # adjust SNR by code rate
snr_adjusted3 = snr_values_dB23  + 10 * np.log10(6)   # adjust SNR by code rate


x_half_1, y_half_1 = run_monte_carlo_simulation(snr_values_dB21 ,modulation.QAMModem(64), 15,puncturing_pattern=None,rate=1/2, K=6)
x_half_2 ,y_half_2 = run_monte_carlo_simulation(snr_values_dB22 ,modulation.QAMModem(64), 15, puncturing_pattern=puncturing_pattern_3_4,rate=3/4,K=6)
x_half_3, y_half_3 = run_monte_carlo_simulation(snr_values_dB23 , modulation.QAMModem(64),15,puncturing_pattern=puncturing_pattern_5_6,rate=5/6,K=6)


plt.semilogy(snr_adjusted1, y_half_1, label='r=1/2-qPSK')
plt.semilogy(snr_adjusted2 , y_half_2, label='r=3/4-qPSK')
plt.semilogy(snr_adjusted3 , y_half_3, label='r=5/6-qPSK')
plt.xticks([0, 5, 10, 15,20,25,30])
#plt.xlim(0, 25)
#plt.ylim(1e-3,1)
plt.title('BER vs. SNRb/Eb')
plt.xlabel('SNRb/Eb (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True)
plt.show()