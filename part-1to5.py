#Part-1(Jathin)
import numpy as np

c = 299792458.0

def compute_derived_constants(lam, theta_i_deg, n_inc=1.0):
  """Computes angular frequency omega, vacuum wavenumber k0, and

  in-plane wavevector component kx.

  Parameters:
  - lam (float): Laser wavelength in meters (e.g., 633e-9 for 633 nm).
  - theta_i_deg (float): Angle of incidence in degrees.
  - n_inc (float): Refractive index of the incident medium (default is 1.0 for
  air).

  Returns:
  - dict: Contains omega, k0, and kx.
  """
  # Speed of light in vacuum (m/s)
  c = 299792458.0

  # Convert angle from degrees to radians
  theta_i = np.radians(theta_i_deg)

  # 1. Angular frequency omega = 2 * pi * c / lambda
  omega = (2.0 * np.pi * c) / lam

  # 2. Vacuum wavenumber k0 = omega / c = 2 * pi / lambda
  k0 = (2.0 * np.pi) / lam

  # 3. In-plane wavevector component kx = k0 * n_inc * sin(theta_i)
  kx = k0 * n_inc * np.sin(theta_i)

  return {"omega": omega, "k0": k0, "kx": kx}

def get_dielectric_tensor(epsilon_d, Q=None, m=None, is_magnetic=False):
  """Builds the 3x3 dielectric tensor for a layer.

  Parameters:
  - epsilon_d (complex or float): Material optical constant (permittivity).
  - Q (complex or float): Voigt magneto-optical parameter.
  - m (tuple or list): Magnetization vector [mx, my, mz].
  - is_magnetic (bool): True for magnetic layers, False for non-magnetic layers.

  Returns:
  - numpy.ndarray: 3x3 complex dielectric tensor.
  """
  if not is_magnetic or Q is None or m is None:
    # For a non-magnetic layer, the tensor is just epsilon_d * I_3x3
    return epsilon_d * np.eye(3, dtype=complex)
  else:
    mx, my, mz = m
    # For a magnetic layer, construct the full tensor
    tensor_matrix = np.array(
        [
            [1.0, 1j * Q * mz, -1j * Q * my],
            [-1j * Q * mz, 1.0, 1j * Q * mx],
            [1j * Q * my, -1j * Q * mx, 1.0],
        ],
        dtype=complex,
    )
    return epsilon_d * tensor_matrix

#Part-2(Athul)

def berreman_matrix(eps,kx,k0):    #HERE THE DIELECTRIC TENSOR(eps) IS THE ONE CALCULATED FROM ABOVE

    xi  = kx/k0  #Normalized wavefn

    eps = np.asarray(eps, dtype=complex)

    # Extracting tensor components
    exx, exy, exz = eps[0, 0], eps[0, 1], eps[0, 2]
    eyx, eyy, eyz = eps[1, 0], eps[1, 1], eps[1, 2]
    ezx, ezy, ezz = eps[2, 0], eps[2, 1], eps[2, 2]


    # Construct Delta matrix elements row by row
    row0 = [-xi * ezx / ezz, 1.0 - (xi ** 2) / ezz, -xi * ezy / ezz, 0.0]
    row1 = [exx - exz * ezx / ezz, -xi * exz/ezz, exy - exz * ezy / ezz, 0.0]
    row2 = [0.0, 0.0, 0.0, 1.0]
    row3 = [eyx - eyz * ezx / ezz, -xi * eyz/ezz, eyy - eyz * ezy / ezz - (xi ** 2), 0.0]

    delta = np.array([row0, row1, row2, row3], dtype=complex)

    return delta

#Part-3
import scipy
from scipy.linalg import expm

def T_total(k0,d,delta):
    return scipy.linalg.expm(1j * k0 * d * delta)  

#Part-4(Rohith)

n_inc = 1.0
n_sub = 1.5
theta_i_deg = 45
lam = 633e-9
d = 20e-9
m = 0,0,1
epsilon_d = -11.49+18.32j
Q = 0.0283+0.0066j

theta_i = np.radians(theta_i_deg)
omega = (2.0 * np.pi * c) / lam
k0 = (2.0 * np.pi) / lam
kx = k0 * n_inc * np.sin(theta_i)

eps = get_dielectric_tensor(epsilon_d, Q, m, is_magnetic=True)
delta = berreman_matrix(eps,kx,k0)

T_total = T_total(k0,d,delta)

theta_t = np.arcsin(n_inc * np.sin(theta_i) / n_sub)

cos_i = np.cos(theta_i)
cos_t = np.cos(theta_t)


L_inc = np.array([
    [0, cos_i],
    [0, n_inc],
    [1, 0],
    [n_inc * cos_i, 0]
], dtype=complex)


L_ref = np.array([
    [0, -cos_i],
    [0, n_inc],
    [1, 0],
    [-n_inc * cos_i, 0]
], dtype=complex)


L_trans = np.array([
    [0, cos_t],
    [0, n_sub],
    [1, 0],
    [n_sub * cos_t, 0]
], dtype=complex)


T_L = T_total @ L_trans

A = np.hstack((L_ref, -T_L))

incident_s = np.array([[1], [0]])

b_s = -L_inc @ incident_s

x_s = np.linalg.solve(A, b_s)


incident_p = np.array([[0], [1]])

b_p = -L_inc @ incident_p

x_p = np.linalg.solve(A, b_p)


r_ss = x_s[0, 0]
r_ps = x_s[1, 0]

r_sp = x_p[0, 0]
r_pp = x_p[1, 0]


R = np.array([
    [r_ss, r_sp],
    [r_ps, r_pp]
])


print("Matrix A:")
print(A)

print("\nSolution for incident s-polarization:")
print(x_s)

print("\nSolution for incident p-polarization:")
print(x_p)

print("\nReflection Matrix R:")
print(R)
