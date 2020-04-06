"""Define AER-like codecs"""
import numpy as np
import struct
from .util import codec
import functools
import itertools
import pdb

@codec(name="aer")
class AER:
    """
    Define encoder and decoder for AER format.

    The first frame is saved as is: matrix of numbers.
    Then the sequence of events is produced.
    
    AER format represents each event as:
        value (1 byte), polarity (1 byte) i (4 bytes), j (4 bytes), t (4 bytes)

    Polarity -- sign of value:
        For positive values: polarity > 0.
        For negative values: polarity = 0.
    """

    event_size = 14
    buffer = bytearray(event_size)

    @classmethod
    def encoder(cls, frames) -> bytearray:
        """
        Yield binary representation of events from sequence of video frames
        for single frame at a time.
        """
        #pdb.set_trace()
        frame_it = iter(frames)
        prev = next(frame_it).copy()

        #Encode resolution and number of frames
        yield struct.pack('>3I', prev.shape[0], prev.shape[1], len(frames))

        #Encode first frame
        yield prev.tobytes()

        for t, frame in enumerate(frame_it):
            diff = np.subtract(frame, prev, dtype='int16')
            result = bytearray()
            for (i, j), value in np.ndenumerate(diff):
                value = int(value)
                if value == 0:
                    continue

                polarity = 0 if value < 0 else 1
                cls._append_event(result, i, j, t, abs(value), polarity)
        
            #Mark end of events for timestamp t
            #result += int(0).to_bytes(1, byteorder='big', signed=False)

            prev = frame.copy()
            if len(result) != 0:
                yield result
    
    @classmethod
    def decoder(cls, data) -> np.ndarray:
        data_it = iter(data)

        header = struct.unpack('>3I', cls.take_next(data_it, 12))
        res = (header[0], header[1])
        frame_num = header[2]

        frame = cls._decode_first_frame(data_it, res) 
        yield frame

        t = 0
        prev_t = 0
        while True:
            try:
                buffer = cls.take_next(data_it, 14)
            except:
                break

            i,j,t,value,polarity = struct.unpack('>3I2B', buffer)
            
            if t > prev_t:
                for _ in range(t-prev_t):
                    yield frame
                prev_t = t
            
            value = value if polarity == 1 else -value
            frame[i,j] += value
        yield frame

        for i in range(frame_num-t-2):
            yield frame

    
    @staticmethod
    def take_next(it, n):
        result = bytearray()
        for _ in range(n):
            result += next(it).to_bytes(1, byteorder='big', signed=False)
        return result
    
    @staticmethod
    def _append_event(result, x, y, t, value, polarity):
        result += x.to_bytes(4, byteorder='big', signed=False)
        result += y.to_bytes(4, byteorder='big', signed=False)
        result += t.to_bytes(4, byteorder='big', signed=False)
        result += value.to_bytes(1, byteorder='big', signed=False)
        result += polarity.to_bytes(1, byteorder='big', signed=False)

    @staticmethod
    def _decode_first_frame(data, res) -> np.ndarray:
        result = np.zeros(res, dtype='uint8')
        for index, _ in np.ndenumerate(result):
            result[index] = next(data)
        return result
        
@codec(name="aer_binary")
class AERBinary:
    """
    Define encoder and decoder for binary AER format.
    
    Binary AER format represents each event as:
        i (4 bytes), j (4 bytes), t (4 bytes), polarity (1 byte)
    
    Polarity signifies the sign of change: 
        255 -- positive change,
        0 -- negative change.
    """

    @classmethod
    def decoder(cls, frames):
        prev = frames.start_frame

        for t, frame in enumerate(frames):
            diff = np.subtract(frame, prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    if value == 0:
                        continue
                    result += i.to_bytes(4, byteorder='big', signed=False)
                    result += j.to_bytes(4, byteorder='big', signed=False)
                    result += t.to_bytes(4, byteorder='little', signed=False)
                    if value >= 0:
                        result += int(255).to_bytes(1, byteorder='big', signed=False)
                    else:
                        result += int(0).to_bytes(1, byteorder='big', signed=False)
            prev = frame
            yield result
    
    @classmethod
    def encoder(cls, events):
        pass

@codec(name="aer_lossy_threshold")
class AERLossy:
    """
    Same as AER format, but lossy.

    AER format represents each event as:
        i (4 bytes), j (4 bytes), t (float), value (1 byte), polarity (1 byte)

    Events with value smaller or equal to the threshold are ignored.
    Consecutive ignored values are accumulated, until their sum is > threshold, in
    which case a new event with value of this sum is created at the current
    timestamp.
    """

    threshold = 5
    
    @classmethod
    def encoder(cls, frames):
        threshold = cls.threshold
        res = frames.conf.res
        ignored_sums = np.zeros((res[0], res[1]))
        prev = frames.start_frame

        for t, frame in enumerate(frames):
            diff = np.subtract(frame, prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    if value == 0:
                        continue
                    elif abs(value) <= threshold:
                        ignored_sums[i][j] += value
                        if abs(ignored_sums[i][j]) > threshold:
                            cls._event_to_bytes(i,j,t+1,
                                    ignored_sums[i][j], result)
                            ignored_sums[i][j] = 0
                    else:
                        ignored_sums[i][j] = 0
                        cls._event_to_bytes(i,j,t+1,value,result)
            prev = frame.copy()
            yield result

    @classmethod
    def decoder(cls, events):
        pass
    
    @classmethod
    def _event_to_bytes(cls, i, j, t, value, result):
        value = int(value)
        result += i.to_bytes(4, byteorder='big', signed=False)
        result += j.to_bytes(4, byteorder='big', signed=False)
        result += bytearray(struct.pack("f", t))

        sign = 0 if value < 0 else 255
        if abs(value) >= 127:
            result += int(127).to_bytes(1, byteorder='big', signed=False)
            overflow = int(abs(value) - 127)
            result += i.to_bytes(4, byteorder='big', signed=False)
            result += j.to_bytes(4, byteorder='big', signed=False)
            #use the same timestamp
            result += bytearray(struct.pack("f", t))
            result += int(sign & overflow).to_bytes(1,
                      byteorder='big', signed=False)
        else:
            polarity = int(sign & abs(value))
            result += polarity.to_bytes(1, byteorder='big', signed=False)