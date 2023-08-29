Physics-Informed Neural Network (PINN) on pendulum system, inspired on the work of https://github.com/benmoseley/harmonic-oscillator-pinn-workshop

The model is set out to:

1. Replicate the differential-equation of a pendulum system $\ddot{\theta} + \dot{\theta}\frac{\mu}{m} + \sin{\theta}\frac{g}{L} = 0$

2. Inverse-training for underlying parameter discovery. The system - specific parameters $\mu, m, L$ can potentially be treated as trainable parameters which converge during training. Finding them simultaneously is more difficult than finding them independently.

3. Etc. 
