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
from channels import awgn
from mode_selector import mod_selector

x = mod_selector(1e-2)
print(x)


def get_modulation_and_rate(snr, df):
    modulation = None
    rate = None

    for index, row in df.iterrows():
        if snr >= row['SNR (dB)']:
            modulation = row['modulation']
            rate = row['Rate']
        else:
            break

    return modulation, rate


# Example usage
desired_snr = 9  # Change this to your desired SNR
mod, rate = get_modulation_and_rate(desired_snr, x)
print(f"For SNR {desired_snr}, Modulation: {mod}, Rate: {rate}")


def adaptive_modulator(m, x):
    global msg
    if m == 'bpsk':
        # Handle BPSK modulation

        bpsk2 = modulation.PSKModem(2)
        msg = bpsk2.modulate(x)

    elif m == '4qam':
        # Handle 4-QAM modulation
        qam4 = modulation.QAMModem(4)
        msg = qam4.modulate(x)

    elif m == '16qam':
        # Handle 16-QAM modulation
        qam16 = modulation.QAMModem(16)
        msg = qam16.modulate(x)

    elif m == '64qam':
        # Handle 64-QAM modulation
        qam16 = modulation.QAMModem(64)
        msg = qam16.modulate(x)

    return msg



def encoder(bits, r):
    puncturing_pattern_3_4 = [1,1,1,0,0,1]
    puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
    global coded_bits
    if r == 'half':
        v = Viterbi(7, [0o133, 0o171], None)
        coded_bits = v.encode(bits)
        pass
    elif r == '3/4':
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_3_4)
        coded_bits = v.encode(bits)
        pass
    elif r == '5/6':
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_5_6)
        coded_bits = v.encode(bits)
        pass
    return coded_bits


def adaptive_demodulator(m,x):
    global msg
    if m == 'bpsk':
        # Handle BPSK modulation

        bpsk2 = modulation.PSKModem(2)
        msg = bpsk2.demodulate(x, demod_type='hard')

    elif m == '4qam':
        # Handle 4-QAM modulation
        qam4 = modulation.QAMModem(4)
        msg = qam4.demodulate(x, demod_type='hard')

    elif m == '16qam':
        # Handle 16-QAM modulation
        qam16 = modulation.QAMModem(16)
        msg = qam16.demodulate(x, demod_type='hard')

    elif m == '64qam':
        # Handle 64-QAM modulation
        qam16 = modulation.QAMModem(64)
        msg = qam16.demodulate(x, demod_type='hard')
    return msg


def decoder(bits, r):
    global decoded_bits
    puncturing_pattern_3_4 = [1,1,1,0,0,1]
    puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
    if r == 'half':
        # Handle half rate
        v = Viterbi(7, [0o133, 0o171], None)
        decoded_bits = v.encode(bits)
        pass
    elif r == '3/4':
        # Handle 3/4 rate
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_3_4)
        decoded_bits = v.encode(bits)
        pass
    elif r == '5/6':
        # Handle 5/6 rate
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_5_6)
        decoded_bits = v.encode(bits)
        pass
    return decoded_bits