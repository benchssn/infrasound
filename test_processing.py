import unittest
import numpy as np
from processing import Settings, isolate, accelerated, summary, response

class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr=2000
        cls.time=np.arange(120*cls.sr)/cls.sr
        cls.input=.2*np.sin(2*np.pi*8*cls.time)+.2*np.sin(2*np.pi*40*cls.time)
    def test_passband_and_rejection(self):
        y=isolate(self.input,self.sr,Settings())[20*self.sr:-20*self.sr]
        t=self.time[20*self.sr:-20*self.sr]
        amplitude=lambda f:2*abs(np.mean(y*np.exp(-2j*np.pi*f*t)))
        self.assertAlmostEqual(amplitude(8),.2,delta=.005)
        self.assertLess(amplitude(40),.001)
    def test_acceleration_frequency_and_duration(self):
        y,rate=accelerated(.2*np.sin(2*np.pi*8*self.time),self.sr,20)
        stats=summary(y,rate,Settings(low_hz=100,high_hz=200))
        self.assertAlmostEqual(stats['dominant_hz'],160,delta=1)
        self.assertAlmostEqual(len(y)/rate,6,places=3)
    def test_normalization_is_optional(self):
        x=.05*np.sin(2*np.pi*8*self.time)
        plain,_=accelerated(x,self.sr,20)
        normal,_=accelerated(x,self.sr,20,True)
        self.assertLess(np.max(np.abs(plain)),.06)
        self.assertAlmostEqual(np.max(np.abs(normal)),.89,places=6)
    def test_zero_phase_response_is_squared(self):
        f,a=response(self.sr,Settings(phase='causal'))
        f,b=response(self.sr,Settings(phase='zero-phase'))
        np.testing.assert_allclose(b,np.maximum(-200,2*a),atol=1e-8)
    def test_invalid_parameters_and_short_input(self):
        for settings in [Settings(low_hz=20),Settings(high_hz=21),Settings(speed=0),Settings(order=0)]:
            with self.assertRaises(ValueError):isolate(self.input,self.sr,settings)
        with self.assertRaises(ValueError):isolate(np.zeros(10),self.sr,Settings())
    def test_silence_remains_finite(self):
        y=isolate(np.zeros(self.sr*30),self.sr,Settings())
        z,_=accelerated(y,self.sr,20,True)
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(np.max(np.abs(z)),0)
        self.assertIsNone(summary(y,self.sr,Settings())['dominant_hz'])

if __name__=='__main__':unittest.main()
