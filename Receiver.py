# xxxxxxxxxxxx Importing Necessary Packages xxxxxxxxxxxx
import matplotlib.pyplot as plt
import numpy as np
import channels
import scipy.interpolate
from scipy import signal
from sk_dsp_comm import digitalcom as dc
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
from mode_selector import mod_selector


# xxxxxxxxxxxxxxx Receiver xxxxxxxxxxxxxxx
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

def decoder(bits, r):
    global decoded_bits
    puncturing_pattern_3_4 = [1,1,1,0,0,1]
    puncturing_pattern_5_6 = [1,1,1,0,1,1,0,1,0,0]
    if rate == 'half':
        # Handle half rate
        v = Viterbi(7, [0o133, 0o171], None)
        decoded_bits = v.encode(bits)
        pass
    elif rate == '3/4':
        # Handle 3/4 rate
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_3_4)
        decoded_bits = v.encode(bits)
        pass
    elif rate == '5/6':
        # Handle 5/6 rate
        v = Viterbi(7, [0o133, 0o171], puncturing_pattern_5_6)
        decoded_bits = v.encode(bits)
        pass
    return decoded_bits
