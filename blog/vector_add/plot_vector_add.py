import matplotlib.pyplot as plt
import numpy as np

# 参数示例
pi = 19.2e12         # FP32 TFLOPS -> FLOPS
beta = 448e9          # DRAM 带宽 Byte/s
I = np.linspace(0, 0.1, 20)  # operational intensity 范围

# memory bound
P_mem = beta * I
# compute bound
P_compute = np.full_like(I, pi)

# vector add 参数
N = 1048576
compute_vec_add = N          # FLOPs (1 add per element)
memory_vec_add = 3*4*N       # 3个内存访问，每个4 Byte
I_vector_add = compute_vec_add / memory_vec_add
P_vector_add = min(pi, beta * I_vector_add)  # Roofline achievable performance

plt.figure(figsize=(8,6))
plt.plot(I, P_mem, 'r-', label='Memory Bound')
# plt.plot(I, P_compute, 'g-', label='Compute Bound')
# plt.axvline(pi/beta, color='b', linestyle='--', label='I_max')

# 标出 vector add
plt.scatter(I_vector_add, P_vector_add, color='k', s=100, zorder=5)
plt.annotate(f'Vector Add\nI={I_vector_add:.2e}\nP={P_vector_add:.2e} FLOP/s',
             xy=(I_vector_add, P_vector_add),
             xytext=(I_vector_add*1.1, P_vector_add*1.2),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel('Operational Intensity [FLOP/Byte]')
plt.ylabel('Attainable Performance [FLOP/s]')
plt.title('Roofline Model')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.xlim([0, max(I)*1.1])
# plt.ylim([0, pi/beta * 0.1*1.1])
breakpoint()
plt.show()
