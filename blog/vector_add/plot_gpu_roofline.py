import matplotlib.pyplot as plt
import numpy as np

# 参数示例
pi = 19.2e12         # FP32 TFLOPS -> FLOPS
beta = 448e9          # DRAM 带宽 Byte/s
I = np.linspace(0, 100, 50)

# memory bound
P_mem = beta * I
# compute bound
P_compute = np.full_like(I, pi)

plt.figure(figsize=(8,6))
plt.plot(I, P_mem, 'r-', label='Memory')
plt.plot(I, P_compute, 'g-', label='Compute')
plt.axvline(pi/beta, color='b', linestyle='--', label='I_max')

plt.xlabel('Operational Intensity [FLOP/Byte]')
plt.ylabel('Attainable Performance [FLOP/s]')
plt.title('Roofline Model')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

