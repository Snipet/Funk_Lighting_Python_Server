import socket
import numpy as np
import threading
import time

localIP = "0.0.0.0"
localPort = 7000
bufferSize = 1024

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

UDPServerSocket.bind((localIP, localPort))


addresses = []

# Create an array of 32-bit floats in numpy with 256 values, all equal to 4.5
sample_freqs = np.zeros(256, dtype=np.float32) + 4.5

def send_sample_freqs(address):
    # Case sample_freqs to a byte buffer (little endian from big endian)
    sample_freqs_bytes = sample_freqs.tobytes()
    # Add a header byte to the front of the buffer
    sample_freqs_bytes = b'\x01' + sample_freqs_bytes
    UDPServerSocket.sendto(sample_freqs_bytes, address)

def diff_thread_send_freqs():
    while True:
        # Get time before sending the sample_freqs array
        start_time = time.time()

        # Wait for the first address to be added to the addresses array
        while len(addresses) == 0:
            time.sleep(0.1)
            pass
        # Send the sample_freqs array to the first address in the addresses array
        for address in addresses:
            send_sample_freqs(address)

        # Get time after sending the sample_freqs array
        end_time = time.time()
        # Wait so the framerate is 30 fps
        if end_time - start_time < 1/30:
            time.sleep(1/30 - (end_time - start_time))

def main_network_thread():
    while True:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        if address not in addresses:
            addresses.append(address)

network_thread = threading.Thread(target=main_network_thread)
network_thread.start()

# # Create a thread to send the sample_freqs array to all addresses in the addresses array
diff_thread = threading.Thread(target=diff_thread_send_freqs)
diff_thread.start()
