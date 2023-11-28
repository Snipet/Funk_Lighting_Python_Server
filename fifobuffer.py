import numpy as np
from collections import deque

class FifoBuffer:
    def __init__(self, total_size, chunk_size):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.buffer = np.empty(total_size, dtype=np.float64)
        self.deque = deque(maxlen=total_size // chunk_size)

    def push(self, data):
        if len(data) != self.chunk_size:
            raise ValueError("Input data size must be equal to chunk size")

        # Remove the oldest chunk if the buffer is full
        if len(self.deque) == self.deque.maxlen:
            oldest_chunk = self.deque.popleft()
            self.buffer[oldest_chunk] = 0.0  # You can customize how to handle the removed data

        # Append the new chunk to the buffer and deque
        start_index = len(self.deque) * self.chunk_size
        self.buffer[start_index:start_index + self.chunk_size] = data
        self.deque.append(slice(start_index, start_index + self.chunk_size))


    def get_buffer(self):
        return self.buffer

# Example usage:
# fifo = FifoBuffer(total_size=2048, chunk_size=256)

# # Simulate receiving a stream of numbers in chunks of 256
# for i in range(10):
#     chunk = np.random.rand(256)
#     fifo.push(chunk)

# # Get the final buffer
# final_buffer = fifo.get_buffer()
# print(final_buffer)