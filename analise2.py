import os
import numpy as np
from matplotlib import pyplot as plt
import utilities

class main():

	def __init__(self, phi0, Nx0,r_max, sinhW, c_final ):
		# define main variables
		self.phi0 = phi0
		self.Nx0 = Nx0
		self.sinhW = sinhW
		self.c_final = c_final
		self.r_max = r_max

		self.w_q = 1.0/3.0
		self.anisotropic_exponent = 2.0 + 3.0*w_q

		self.sinhA = r_max
		self.inv_sinhW         = 1.0/sinhW;
		self.sinh_inv_W        = np.sinh(inv_sinhW);
		self.A_over_sinh_inv_W = sinhA / sinh_inv_W;
		self.x0_min = 0.0
		self.x0_max = 1.0
		self.dx0 = (x0_max - x0_min)/(Nx0-1.0);
		self.Ngz0 = 0
		self.inv_dx0 = 1.0/dx0



execution = main(0.3, 320,0.2,0)