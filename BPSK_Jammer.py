# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:49:58 2025

@author: EMOPFOR
"""

import numpy as np
import matplotlib.pyplot as plt
import struct

def generate_random_bpsk_bits(num_bits):
    """
    Generates a list of random BPSK bits (0 or 1).

    Args:
        num_bits (int): The number of bits to generate.

    Returns:
        list: A list containing the random bits.
    """
    # Use numpy.random.randint to efficiently generate an array of random integers
    # that are either 0 or 1.  This is much faster than using a Python loop.
    random_bits = np.random.randint(0, 2, num_bits).tolist()  # Convert to a list
    return random_bits

if __name__ == '__main__':
    # Generate 8 random BPSK bits
    num_bits = 8
    random_bits = generate_random_bpsk_bits(num_bits)

    # Print the generated bits
    print(f"Generated {num_bits} random BPSK bits: {random_bits}")

def generate_bpsk_signal(bits, sampling_rate, carrier_frequency=1000):
    """
    Generates a BPSK signal.  This is the same function as before, included
    for completeness.

    Args:
        bits (list): A list of bits (0 or 1).
        sampling_rate (int): The sampling rate of the signal in Hz.
        carrier_frequency (int): The carrier frequency of the signal in Hz.  Defaults to 1000 Hz.

    Returns:
        tuple: A tuple containing:
            - time (numpy.ndarray): The time vector for the signal.
            - signal (numpy.ndarray): The BPSK signal.
    """
    bits = np.array(random_bits)
    bit_duration = 1 / (sampling_rate / len(bits))
    num_samples = len(bits) * sampling_rate //len(bits)
    time = np.arange(0, num_samples) / sampling_rate
    carrier = np.cos(2 * np.pi * carrier_frequency * time)
    symbols = 2 * bits - 1
    signal = np.repeat(symbols, sampling_rate // len(bits)) * carrier
    return time, signal

def generate_jamming_signal(time, carrier_frequency=1000, signal_power=1.0):
    """
    Generates a jamming signal.  This can be a simple sine wave at the
    same carrier frequency as the BPSK signal, but with a random phase.

    Args:
        time (numpy.ndarray): The time vector for the signal.  This should be
            the same time vector as the BPSK signal you want to jam.
        carrier_frequency (int): The carrier frequency of the signal in Hz.
            Defaults to 1000 Hz.  This should match the BPSK signal's
            carrier frequency.
        signal_power (float): The power of the jamming signal.  This allows
            you to control how strong the jamming signal is.  Defaults to 1.0.

    Returns:
        numpy.ndarray: The jamming signal.
    """
    # Generate a sine wave with a random phase.
    phase = np.random.uniform(0, 2 * np.pi)
    jamming_signal = signal_power * np.cos(2 * np.pi * carrier_frequency * time + phase)
    return jamming_signal
    
    
    #jamming_signal = signal_power * np.cos(2 * np.pi * carrier_frequency * time)
    #return jamming_signal


def combine_signals(signal1, signal2):
    """
    Combines two signals by adding them.  This simulates the effect of the
    jamming signal interfering with the BPSK signal.

    Args:
        signal1 (numpy.ndarray): The first signal (e.g., the BPSK signal).
        signal2 (numpy.ndarray): The second signal (e.g., the jamming signal).

    Returns:
        numpy.ndarray: The combined signal.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length to be combined.")
    return signal1 + signal2

if __name__ == '__main__':
    # Example usage:
    #bits = [0, 1, 0, 1, 1, 0, 0, 1]
    sampling_rate = 32000
    carrier_frequency = 1000  # Use the same carrier frequency as the BPSK signal

    time, bpsk_signal = generate_bpsk_signal(random_bits, sampling_rate, carrier_frequency)
    jamming_signal = generate_jamming_signal(time, carrier_frequency, signal_power=0.5) # Adjust signal_power

    # Combine the signals to simulate jamming
    combined_signal = combine_signals(bpsk_signal, jamming_signal)

    # Print first 10 values
    print(f"BPSK Signal (first 10 samples): {bpsk_signal[:10]}")
    print(f"Jamming Signal (first 10 samples): {jamming_signal[:10]}")
    print(f"Combined Signal (first 10 samples): {combined_signal[:10]}")

    # Print the lengths of the signals to verify they are the same.
    print(f"Length of BPSK signal: {len(bpsk_signal)}")
    print(f"Length of Jamming signal: {len(jamming_signal)}")
    print(f"Length of Combined signal: {len(combined_signal)}")
    # You can now analyze or plot the 'combined_signal' to see the effect of the jamming.
    # Again,  if you have matplotlib, you can uncomment and use this code.
    


    plt.figure(figsize=(10, 6))
    plt.plot(time, bpsk_signal, label='BPSK Signal', alpha=0.7)
    plt.plot(time, jamming_signal, label='Jamming Signal', alpha=0.7)
    plt.plot(time, combined_signal, label='Combined Signal (Jammed)', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('BPSK Signal with Jamming')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
def write_iq_file(filename, signal):
    """
    Writes a signal to a .iq file.  The signal is assumed to be real-valued.
    The real part is written as the I component, and the imaginary part is zero.

    Args:
        filename (str): The name of the .iq file to write.
        signal (numpy.ndarray): The signal to write.  This should be a 1-D
            numpy array of real values.
    """
    with open(filename, 'wb') as f:
        for sample in signal:
            # Pack the real and imaginary parts as two 32-bit floats (complex64).
            # The imaginary part is set to 0.
            f.write(struct.pack('<ff', sample, 0))    
    
    # Write the BPSK signal to a .iq file
write_iq_file('bpsk_signal.iq', bpsk_signal)
print("BPSK signal written to bpsk_signal.iq")

# Write the jamming signal to a .iq file
write_iq_file('jamming_signal.iq', jamming_signal)
print("Jamming signal written to jamming_signal.iq")

# Write the combined signal to a .iq file
write_iq_file('combined_signal.iq', combined_signal)
print("Combined signal written to combined_signal.iq")