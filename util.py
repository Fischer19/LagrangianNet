import numpy as np
import torch

def gen_paraboloid(t, v_0, m = 1, g = 10):
    h = v_0 * t - 1/2 * g * t ** 2
    v = v_0 - g * t
    return h

def compute_action(m, g, H, delta_t):
    A = 0
    L = []
    for i, h in enumerate(H[1:]):
        K = 1/2 * m * ((h - H[i]) / delta_t) ** 2
        P = m * g * h
        L.append(K - P)
        A += (K - P)
    return A, L

def gen_random_path(T = 100, t_0 = 0, t_n = 5):
    def f(x, n):
        return (x - t_0) * (x - n) * (x - t_n)
    P = []
    n = np.random.randn() * (t_n - t_0)
    for i in range(T):
        t = i * (t_n - t_0) / T
        P.append(f(t, n))
    return P

def gen_training_data(size, len, dt = 0.01):
    X = []
    V = []
    for i in range(size):
        h = []
        v = []
        v_0 = np.random.uniform(0.1, 10)
        for j in range(len + 1):
            t = j * dt
            h_now = gen_paraboloid(t, v_0)
            h.append(h_now)
            if j > 0:
                v.append((h[-1] - h[-2]) / dt)
        x = []
        for j in range(len):
            x.append([h[j], v[j]])
        X.append(x)
        
    return np.array(X)
    
def f(x):
    return torch.log(x + 1e-10)**2

def Euler_Lagrange(model, y0, y1, dt):
    L_0 = model(y0)
    dLdq_0 = model.derivatives(y0)[0]
    L_1 = model(y1)
    dLdq_1 = model.derivatives(y1)[0]
    return ((dLdq_1[1] - dLdq_0[1]) / dt - dLdq_0[0]) ** 2

def Euler_Lagrange_normalizer(model, y0, y1, dt):
    L_0 = model(y0)
    dLdq_0 = model.derivatives(y0)[0]
    L_1 = model(y1)
    dLdq_1 = model.derivatives(y1)[0]
    return ((dLdq_1[1] - dLdq_0[1]) / dt) ** 2 + dLdq_0[0] ** 2

def L2_loss(u, v):
    return (u-v).pow(2).mean()

def variation_integrate(Qmodel, model, x0, x1, t, dt):
    integral = 0
    start0, end0 = x0[0].detach().item(), x1[0].detach().item()
    start1, end1 = x0[1].detach().item(), x1[1].detach().item()
    mesh0 = torch.linspace(start0, end0, t)
    mesh1 = torch.linspace(start1, end1, t)
    mesh = torch.cat((mesh0.view(-1, 1), mesh1.view(-1,1)), axis = 1)
    for x in mesh:
        integral += model(Qmodel(x.double())) * dt / t
    return integral


def predict(index):
	result = []
	for x in X_train[index]:
	    l = model(x)
	    result.append(l)
	return result
