import numpy as np
import matplotlib.pyplot as plt
from pyphysim.modulators.fundamental import BPSK, QAM, Modulator, QPSK
import channels

# Parameters
N = 10**6 # Number of bits to simulate
SNR_range = range(0, 31)  # Range of SNR in dB
thresholds = 5 * 10**-3  # BER threshold
M1 = 2  # BPSK 
M2 = 4  # QPSK
M3 = 16  # 16QAM

def calculate_bit_error_rate(original_data, received_data):
    num_errors = np.sum(original_data != received_data)
    bit_error_rate = num_errors / len(original_data)
    return bit_error_rate

# Generate random binary data
x1 = np.random.randint(0, M1, N)  # BPSK signal
x2 = np.random.randint(0, M2, N)  # QPSK signal
x3 = np.random.randint(0, M3, N)  # 16QAM signal

# Modulation
bpsk = BPSK()
qpsk = QPSK()
qam16 = QAM(16)
h1 = bpsk.modulate(x1)  # BPSK modulation
h2 = qpsk.modulate(x2)  # QPSK modulation (modulated symbol)
h3 = qam16.modulate(x3) # 16QAM modulation(modulated symbol)

# Generate Rayleigh fading channel
R = np.random.rayleigh(scale=0.5,size=N) 

# Apply Rayleigh fading
H1 = h1 * R  # BPSK with Rayleigh Channel
H2 = h2 * R  # QPSK with Rayleigh Channel
H3 = h3 * R  # 16QAM with Rayleigh Channel

# BER vs SNR Simulation
BPSK_AWGN = []
QPSK_AWGN = []
QAM_AWGN = []
BPSK_Ray = []
QPSK_Ray = []
QAM_Ray = []
BPSK_Ray_Equalize = []
QPSK_Ray_Equalize = []
QAM_Ray_Equalize = []

for snr in SNR_range:
    yAn1 = channels.awgn(h1, snr)   # Channel (Additive White Gaussian Noise)
    yA1 = bpsk.demodulate(yAn1) # Demodulation
    BPSK_AWGN.append(calculate_bit_error_rate(x1, yA1)) # Calculate BER

    yAn2 = channels.awgn(h2, snr)   # Channel (Additive White Gaussian Noise)
    yA2 = qpsk.demodulate(yAn2) # Demodulation
    QPSK_AWGN.append(calculate_bit_error_rate(x2, yA2)) # Calculate BER

    yAn3 = channels.awgn(h3, snr)   # Channel (Additive White Gaussian Noise)
    yA3 = qam16.demodulate(yAn3) # Demodulation
    QAM_AWGN.append(calculate_bit_error_rate(x3, yA3)) # Calculate BER

    yRn1 = channels.awgn(H1, snr)   # Channel (Additive White Gaussian Noise)
    yR1 = bpsk.demodulate(yRn1) # Demodulation
    BPSK_Ray.append(calculate_bit_error_rate(x1, yR1)) # Calculate BER

    yRn2 = channels.awgn(H2, snr)   # Channel (Additive White Gaussian Noise)
    yR2 = qpsk.demodulate(yRn2) # Demodulation
    QPSK_Ray.append(calculate_bit_error_rate(x2, yR2)) # Calculate BER

    yRn3 = channels.awgn(H3, snr)   # Channel (Additive White Gaussian Noise)
    yR3 = qam16.demodulate(yRn3) # Demodulation
    QAM_Ray.append(calculate_bit_error_rate(x3, yR3)) # Calculate BER

    yREn1 = yRn1 / R     # Equalize rayleigh channel
    yRE1 = bpsk.demodulate(yREn1)   #Demodulation
    BPSK_Ray_Equalize.append(calculate_bit_error_rate(x1, yRE1))

    yREn2 = yRn2 / R    # Equalize rayleigh channel
    yRE2 = qpsk.demodulate(yREn2)   #Demodulation
    QPSK_Ray_Equalize.append(calculate_bit_error_rate(x2, yRE2))

    yREn3 = yRn3 / R   # Equalize rayleigh channel
    yRE3 = qam16.demodulate(yREn3)   #Demodulation
    QAM_Ray_Equalize.append(calculate_bit_error_rate(x3, yRE3))
    
# Plot BER vs SNR
plt.figure(1)
plt.semilogy(SNR_range, BPSK_AWGN, marker='o', linestyle='-', label='BPSK_AWGN')
plt.semilogy(SNR_range, QPSK_AWGN, marker='o', linestyle='-', label='QPSK_AWGN')
plt.semilogy(SNR_range, QAM_AWGN, marker='o', linestyle='-', label='16-QAM_AWGN')
plt.semilogy(SNR_range, BPSK_Ray, marker='*', linestyle='dotted', label='BPSK_Rayleigh')
plt.semilogy(SNR_range, QPSK_Ray, marker='*', linestyle='dotted', label='QPSK_Rayleigh')
plt.semilogy(SNR_range, QAM_Ray, marker='*', linestyle='dotted', label='QAM_Rayleigh')
# plt.semilogy(SNR_range, BPSK_Ray_Equalize,marker='d', linestyle='dashed', label='BPSK_Ray_Equalize')
# plt.semilogy(SNR_range, QPSK_Ray_Equalize, marker='d', linestyle='dashed', label='QPSK_Ray_Equalize')
# plt.semilogy(SNR_range, QAM_Ray_Equalize, marker='d', linestyle='dashed', label='QAM_Ray_Equalize')
plt.grid(True)
plt.axis([-1, 30, 10**-5, 1.2])
plt.plot([0, 30], [thresholds, thresholds], color='red', linestyle='--')
plt.title('BER vs SNR for BPSK, QPSK, 16-QAM for AWGN, AWGN+Rayleigh, Rayleigh equalize')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()

plt.figure(2)
plt.semilogy(SNR_range, BPSK_Ray_Equalize, ':rx')
plt.semilogy(SNR_range, QPSK_Ray_Equalize, ':gx')
plt.semilogy(SNR_range, QAM_Ray_Equalize, ':bx')
plt.grid(True)
plt.axis([-1, 30, 10**-5, 1.2])
plt.plot([0, 30], [thresholds, thresholds], color='red', linestyle='--')
plt.legend(['BPSK Ray Equalize', 'QPSK Ray Equalize', '16QAM Ray Equalize'])
plt.title('BER vs SNR in BPSK&QPSK&16QAM Ray Equalized')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')

plt.show()