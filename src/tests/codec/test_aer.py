import pytest
import src.event_compression.synthetic as synthetic
from src.event_compression.synthetic.sequence import *
import src.event_compression.codec as codec
from src.event_compression.codec import aer as aer
import itertools
import functools
import numpy as np
import pdb
import struct
import copy

def simple_seq():
    """
    Produce sequence with events (x, y, t, v, p):
        (1, 1, 1, 255, +)
        (1, 1, 3, 248, -)
        (0, 0, 4, 255, +)
        (1, 1, 4, 7,   -)
    """
    return [
    np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype='uint8'),
    np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype='uint8'),
    np.array([
        [0, 0, 0],
        [0, 255, 0],
        [0, 0, 0]
    ], dtype='uint8'),
    np.array([
        [0, 0, 0],
        [0, 255, 0],
        [0, 0, 0]
    ], dtype='uint8'),
    np.array([
        [0, 0, 0],
        [0, 7, 0],
        [0, 0, 0]
    ], dtype='uint8'),
    np.array([
        [255, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype='uint8')]

def reduce(encoder):
    return functools.reduce(lambda x, y: x+y, encoder, bytearray())

def common_test(cls=None):
    """
    Define common tests for all aer codecs.

    @common_test(cls=aer_class_to_be_tested)
    class TestClass:
        pass
    """
    def decorate(_cls):
        _cls.cls = cls

        def test_invertability(instance, sequence, correct):
            code = reduce(instance.cls.encoder(sequence))
            decoder = instance.cls.decoder(code)
            frames = [x.copy() for x in decoder]
            for i, f in enumerate(frames):
                np.testing.assert_equal(f, correct[i])
        
        #Add all invertability tests
        for name, sequence in synthetic.sequences().items():
            config = Config((3,3),10,1,value=127,rate=0.1)
            np.random.seed(seed=0)
            correct_seq = [x.copy() for x in sequence(config)]
            seq = copy.deepcopy(correct_seq)
            def caller(seq, correct_seq):
                return lambda inst: test_invertability(inst,seq,correct_seq)
            setattr(_cls, f'test_invertability_{name}', caller(seq, correct_seq))
        fn = lambda inst: test_invertability(inst, simple_seq(), simple_seq())
        setattr(_cls, f'test_{cls.__name__}_invertability_simple', fn)
        
        return _cls
    return decorate

def test_init():
	assert codec.codecs()

@common_test(aer.AER)
class TestAER:
    def test_encoder(self):
        sequence = simple_seq()
        code = reduce(self.cls.encoder(sequence))
        correct_code = struct.pack('>3I 9B 3I2B 3I2B 3I2B 3I2B',
            3,3,6,
            0,0,0,0,0,0,0,0,0,
            1,1,1,255,1,
            1,1,3,248,0,
            0,0,4,255,1,
            1,1,4,7,0)
        assert code == correct_code

    def test_encode_moving_edge(self):
        """
        [[0 0 0]
        [0 0 0]
        [0 0 0]]

        [[255   0   0]
        [255   0   0]
        [255   0   0]]

        [[255 255   0]
        [255 255   0]
        [255 255   0]]

        [[255 255   0]
        [255 255   0]
        [255 255   0]]
        """
        #pdb.set_trace()
        seq = MovingEdge(Config((3,3), 2, 2))
        code = reduce(self.cls.encoder(seq))
        correct_code = struct.pack('>3I 9B 3I2B 3I2B 3I2B 3I2B 3I2B 3I2B',
            3,3,4,
            0,0,0,0,0,0,0,0,0,
            
            0,0,0,255,1,
            1,0,0,255,1,
            2,0,0,255,1,

            0,1,1,255,1,
            1,1,1,255,1,
            2,1,1,255,1)
        
        assert code == correct_code
    
    def test_encode_single_color(self):
        seq = SingleColor(Config((3,3),4,1,value=127))
        code = reduce(self.cls.encoder(seq))
        correct_code = struct.pack('>3I 9B',
            3,3,4,
            127,127,127,127,127,127,127,127,127)
        assert code == correct_code

    def test_decode_single_color(self):
        seq = [x for x in SingleColor(Config((3,3),4,1,value=127))]
        code = struct.pack('>3I 9B',
            3,3,4,
            127,127,127,127,127,127,127,127,127)
        frames = [x.copy() for x in self.cls.decoder(code)]
        for i, frame in enumerate(frames):
            np.testing.assert_equal(frame, seq[i])

