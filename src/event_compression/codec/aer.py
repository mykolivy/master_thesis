"""Define AER-like codecs"""
import numpy as np
import struct
from . import codec

@codec(name="aer")
class AER:
    """
    Define encoder and decoder for AER format.
    
    AER format represents each event as:
        i (4 bytes), j (4 bytes), t (float), value (1 byte), polarity (1 byte)
    
    If value >= 127, two events are created at the same timestamp (127 and
    residual), so that each value can be stored with one byte.

    Polarity -- sign of value:
        For positive values: polarity > 0.
        For negative values: polarity = 0.
    """

    @classmethod
    def encoder(cls, frames) -> bytearray:
        prev = frames.start_frame
        overflow = bytearray()

        for t, frame in enumerate(frames):
            diff = np.subtract(frame, prev)
            result = bytearray()
            for i, row in enumerate(diff):
                for j, value in enumerate(row):
                    value = int(value)
                    if value == 0:
                        continue
                    result += i.to_bytes(4, byteorder='big', signed=False)
                    result += j.to_bytes(4, byteorder='big', signed=False)
                    result += bytearray(struct.pack("f", t))

                    sign = 0 if value < 0 else 255
                    if abs(value) >= 127:
                        result += int(127).to_bytes(1, byteorder='big', signed=False)
                        overflow = int(abs(value) - 127)
                        result += i.to_bytes(4, byteorder='big', signed=False)
                        result += j.to_bytes(4, byteorder='big', signed=False)
                        result += bytearray(struct.pack("f", t))
                        result += int(sign & overflow).to_bytes(1,
                                  byteorder='big', signed=False)
                    else:
                        result += int(sign & abs(value)).to_bytes(1,
                                byteorder='big', signed=False)
            prev = frame
            yield result
    
    @classmethod
    def decoder(cls, events):
        pass

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