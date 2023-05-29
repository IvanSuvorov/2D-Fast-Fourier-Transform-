import cmath
from numba import njit
import numpy as np


"""
Use fft for 2D FFT
other functions are technical
"""


def fft2d(signal):
    """
    Example of 2D FFT realisation by rows and columns
    Computes the 2D FFT of an input array x
    Further we will use it as a reference
    """
    # Compute the FFT of each row
    fft_by_rows = np.fft.fft(signal, axis=1)
    # Compute the FFT of each column
    result = np.fft.fft(fft_by_rows, axis=0)
    return result

# Example usage


x = np.random.rand(4, 4)  # create a random 4x4 array
y = fft2d(x)              # compute the 2D FFT of x
# print(y)


# next decorator applies numba
# optimization for particular function
# https://numba.readthedocs.io/en/stable/user/index.html


@njit
def recursion_base(signal):
    """
    here we calculate F(p, m)
    for 2*2-sized signal, using
    convenient values of
    e^((2*pi/2^s)(k*p + t*m))
    at s = 1,  k,p,t,m = 0,1
    """
    f00 = signal[0][0] + signal[0][1] + signal[1][0] + signal[1][1]
    f01 = signal[0][0] - signal[0][1] + signal[1][0] - signal[1][1]
    f10 = signal[0][0] + signal[0][1] - signal[1][0] - signal[1][1]
    f11 = signal[0][0] - signal[0][1] - signal[1][0] + signal[1][1]
    answer = [[f00, f01], [f10, f11]]
    return answer


@njit
def w(F, p, s):
    """
    2^s-th root of unity to the power of p
    """
    return F * cmath.exp(2 * cmath.pi * 1j * p / pow(2, s))


@njit
def single_coordinate_ft(signal, signal_size_power, p, m, k, t):
    return signal[k][t] * cmath.exp(2 * cmath.pi * (k * p + t * m) / pow(2, signal_size_power))


@njit
def fft_back(signal, signal_size_power):
    """
    this is recurrent function for fft
    """
    if signal_size_power == 1:
        return recursion_base(signal)
    result = [[0 for i in range(pow(2, signal_size_power))] for j in range(pow(2, signal_size_power))]
    signal00 = [[0 for i in range(pow(2, signal_size_power - 1))] for j in range(pow(2, signal_size_power - 1))]
    signal01 = [[0 for i in range(pow(2, signal_size_power - 1))] for j in range(pow(2, signal_size_power - 1))]
    signal10 = [[0 for i in range(pow(2, signal_size_power - 1))] for j in range(pow(2, signal_size_power - 1))]
    signal11 = [[0 for i in range(pow(2, signal_size_power - 1))] for j in range(pow(2, signal_size_power - 1))]
    for p1 in range(0, pow(2, signal_size_power - 1)):
        for m1 in range(0, pow(2, signal_size_power - 1)):
            signal00[p1][m1] = signal[2 * p1][2 * m1]
            signal01[p1][m1] = signal[2 * p1][2 * m1 + 1]
            signal10[p1][m1] = signal[2 * p1 + 1][2 * m1]
            signal11[p1][m1] = signal[2 * p1 + 1][2 * m1 + 1]

    result00 = fft_back(signal00, signal_size_power - 1)
    result01 = fft_back(signal01, signal_size_power - 1)
    result10 = fft_back(signal10, signal_size_power - 1)
    result11 = fft_back(signal11, signal_size_power - 1)
    for p1 in range(0, pow(2, signal_size_power - 1)):
        for m1 in range(0, pow(2, signal_size_power - 1)):
            result[2*p1][2*m1] = result00[p1][m1]
            result[2 * p1][2 * m1 + 1] = result01[p1][m1] * w(result01[p1][m1], p1, signal_size_power)
            result[2 * p1 + 1][2 * m1] = result10[p1][m1] * w(result10[p1][m1], m1, signal_size_power)
            result[2 * p1 + 1][2 * m1 + 1] = result11[p1][m1] * w(result11[p1][m1], p1 + m1, signal_size_power)
    return result


@njit
def fft(signal):
    """
    this is interface for previous function, use it
    """
    s = int((cmath.log(len(signal), 2)).real)
    return fft_back(signal, s)
