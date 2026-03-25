"""Time-varying objective function classes; all inherit ObjectiveFunction and expose update, eval, gradient, get_objective_info."""

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ObjectiveFunction(ABC):
    def __init__(self, nx):
        self.nx = nx
        self.dt = 1

    @abstractmethod
    def eval(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def get_objective_info(self):
        pass

    @abstractmethod
    def update(self, t):
        pass

    def estimate_sector_variation(self, T_probe=100):
        """Scan 200 time steps and return (L_min, L_max, m_min, m_max, dL_max, dm_max, dL_min, dm_min)."""
        T_probe = 200

        _, m_0, L_0 = self.get_objective_info()
        delta_L_max = -np.inf
        delta_m_max = -np.inf
        delta_L_min =  np.inf
        delta_m_min =  np.inf
        m_prev, L_prev = m_0, L_0
        m_min, m_max   = m_0, m_0
        L_min, L_max   = L_0, L_0

        is_robotic = (hasattr(self, 'name')
                    and self.name == 'robotic_ellipse_tracking')

        for k in range(1, T_probe + 1):
            self.update(k)
            if is_robotic:
                np.random.seed(42 + k)
                x_sim = 0.1 * np.random.randn(self.nx, 1)
                self.update_joint_angles(x_sim)
            _, m_k, L_k = self.get_objective_info()

            delta_m_max = max(delta_m_max, m_k - m_prev)
            delta_L_max = max(delta_L_max, L_k - L_prev)
            delta_m_min = min(delta_m_min, m_k - m_prev)
            delta_L_min = min(delta_L_min, L_k - L_prev)
            m_min = min(m_min, m_k)
            m_max = max(m_max, m_k)
            L_min = min(L_min, L_k)
            L_max = max(L_max, L_k)
            m_prev, L_prev = m_k, L_k

        return L_min, L_max, m_min, m_max, delta_L_max, delta_m_max, delta_L_min, delta_m_min


# ---------------------------------------------------------------------------
# Periodic 2-D example
# ---------------------------------------------------------------------------

class PeriodicExample2D(ObjectiveFunction):

    def __init__(self, omega):
        super().__init__(nx=2)
        self.name      = 'periodic_example_2d'
        self.omega     = omega
        self.current_t = 0

    def update(self, t):
        self.current_t = t

    def eval(self, x, prev_t=False):
        t = self.current_t if not prev_t else self.current_t - 1
        x = x.reshape(-1, self.nx)
        f = ((x[:, 0] - np.exp(np.cos(self.omega * t)))**2
             + (x[:, 1] - x[:, 0] * np.tanh(np.sin(self.omega * t)))**2)
        return f

    def gradient(self, x):
        t   = self.current_t
        if x.ndim > 1:
            x = x.reshape(-1, self.nx)
        a   = np.exp(np.cos(self.omega * t))
        tau = np.tanh(np.sin(self.omega * t))
        g0  = 2 * (x[:, 0] - a) - 2 * tau * (x[:, 1] - x[:, 0] * tau)
        g1  = 2 * (x[:, 1] - x[:, 0] * tau)
        return np.column_stack([g0, g1]).reshape(-1, 1)

    def hessian(self, x):
        t = self.current_t
        H = np.zeros((2, 2))
        H[0, 0] = 2 + 2 * np.tanh(np.sin(self.omega * t))**2
        H[0, 1] = -2 * np.tanh(np.sin(self.omega * t))
        H[1, 0] = H[0, 1]
        H[1, 1] = 2
        return H

    def get_objective_info(self):
        t = self.current_t
        x_star = np.array([
            [np.exp(np.cos(self.omega * t))],
            [np.exp(np.cos(self.omega * t)) * np.tanh(np.sin(self.omega * t))],
        ])
        m, L = self._sector_constraints(t)
        return x_star, m, L

    def _sector_constraints(self, t):
        tau = np.tanh(np.sin(self.omega * t))
        A   = np.array([[2 + 2 * tau**2, -2 * tau],
                        [-2 * tau,        2]])
        eigs = np.linalg.eigvals(A)
        return float(min(eigs)), float(max(eigs))


# ---------------------------------------------------------------------------
# Windowed least squares
# ---------------------------------------------------------------------------

class WindowedLeastSquares(ObjectiveFunction):

    def __init__(self, nx=10, n_data=100, noise=0.1):
        super().__init__(nx)
        self.name   = 'windowed_least_squares'
        self.n_data = n_data
        self.noise  = noise
        self.At_list = []
        self.bt_list = []
        self._initialize_window()

    def _initialize_window(self):
        nx, n_data = self.nx, self.n_data
        for t in range(-n_data, 0):
            A0  = -1 + 2 * np.random.rand(n_data, nx)
            b0  = np.zeros((n_data, 1))
            At  = 0.5 * np.sin(0.1 * t) + A0 + self.noise * np.random.randn(n_data, nx)
            bt  = 1.5 * np.sin(0.1 * t) + b0 + self.noise * np.random.randn(n_data, 1)
            self.At_list.append(At)
            self.bt_list.append(bt)

    def update(self, t):
        nx, n_data = self.nx, self.n_data
        A0  = -1 + 2 * np.random.rand(n_data, nx)
        b0  = np.zeros((n_data, 1))
        new_At = 0.5 * np.sin(0.1 * t) + A0 + self.noise * np.random.randn(n_data, nx)
        new_bt = 1.5 * np.sin(0.1 * t) + b0 + self.noise * np.random.randn(n_data, 1)
        self.At_list.append(new_At)
        self.bt_list.append(new_bt)
        if len(self.At_list) > n_data:
            self.At_list.pop(0)
            self.bt_list.pop(0)
        self._AtA = sum(A.T @ A for A in self.At_list)
        self._Atb = sum(A.T @ b for A, b in zip(self.At_list, self.bt_list))

    def eval(self, x, prev_t=False):
        x_star = self._optimum()
        f = 0.5 * sum(np.linalg.norm(self.At_list[i] @ x - self.bt_list[i])**2
                      for i in range(len(self.At_list)))
        f_star = 0.5 * sum(np.linalg.norm(self.At_list[i] @ x_star - self.bt_list[i])**2
                           for i in range(len(self.At_list)))
        return np.array([f - f_star])

    def gradient(self, x):
        return self._AtA @ x - self._Atb

    def get_objective_info(self):
        x_star = self._optimum()
        m, L   = self._sector_constraints()
        return x_star, m, L

    def _optimum(self):
        return np.linalg.solve(self._AtA, self._Atb)

    def _sector_constraints(self):
        eigs = np.linalg.eigvals(self._AtA)
        return float(min(eigs)), float(max(eigs))


# ---------------------------------------------------------------------------
# Network cos-log problem
# ---------------------------------------------------------------------------

class NetworkCosLog(ObjectiveFunction):

    def __init__(self, N=20, A=2.5, omega=np.pi / 80, b=0.0):
        super().__init__(nx=1)
        self.name      = 'network_cos_log'
        self.N         = N
        self.A         = A
        self.omega     = omega
        self.b         = b
        self.current_t = 0
        self.a_i   = np.random.uniform(-10, 10, N)
        self.phi_i = np.random.uniform(0, 2 * np.pi, N)

    def update(self, t):
        self.current_t = t

    def eval(self, x, prev_t=False):
        t     = self.current_t if not prev_t else self.current_t - 1
        term1 = 0.5 * np.sum((x - self.A * np.cos(self.omega * t + self.phi_i - self.b * t))**2)
        term2 = np.sum(np.log(1 + np.exp(x - self.a_i)))
        return np.array([term1 + term2])

    def gradient(self, x):
        x    = np.atleast_1d(x).astype(float)
        t    = self.current_t
        grad = np.sum(
            (x - self.A * np.cos(self.omega * t + self.phi_i - self.b * t))
            + (np.exp(x - self.a_i) / (1 + np.exp(x - self.a_i)))
        )
        return np.array([[grad]])

    def get_objective_info(self):
        t      = self.current_t
        m, L   = self.N, self.N * 1.25
        x_star = np.mean(self.A * np.cos(self.omega * t + self.phi_i - self.b * t))
        return np.array([[x_star]]), m, L


# ---------------------------------------------------------------------------
# Time-varying QP with log-barrier
# ---------------------------------------------------------------------------

class TVQPBarrier(ObjectiveFunction):
    """Time-varying quadratic plus log-barrier objective with exponentially growing barrier weight."""

    def __init__(self):
        super().__init__(nx=2)
        self.name      = 'tvqp_barrier'
        self.current_t = 0

    def c_t(self, t): return 10.0 * np.exp(t)
    def s_t(self, t): return 2.0  * np.exp(-5.0 * t)

    def update(self, t):
        self.current_t = t

    def _affine_gap(self, x, t):
        return self.s_t(t) + np.cos(t) + x[0] - x[1]

    def eval(self, x, prev_t=False):
        t   = self.current_t if not prev_t else self.current_t - 1
        x   = np.asarray(x).reshape(2,)
        gap = self._affine_gap(x, t)
        if gap <= 0:
            return np.inf
        c = self.c_t(t)
        f = (0.5*(x[0] + np.sin(t))**2
             + 1.5*(x[1] + np.cos(t))**2
             - (1.0/c) * np.log(gap))
        return np.array([[f]])

    def gradient(self, x):
        t   = self.current_t
        x   = np.asarray(x).reshape(2,)
        c   = self.c_t(t)
        gap = self._affine_gap(x, t)
        g1  = (x[0] + np.sin(t)) - (1.0/c)*(1.0/gap)
        g2  = 3.0*(x[1] + np.cos(t)) + (1.0/c)*(1.0/gap)
        return np.array([[g1], [g2]])

    def hessian(self, t, x):
        x   = np.asarray(x).reshape(2,)
        c   = self.c_t(t)
        gap = self._affine_gap(x, t)
        B   = (1.0/c) * (1.0/(gap**2))
        return (np.array([[1.0, 0.0], [0.0, 3.0]])
                + B * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    def get_objective_info(self):
        t     = self.current_t
        x_ref = np.array([[-np.sin(t)], [-np.cos(t)]])
        m, L  = self._sector_constraints(t, x_ref)
        return x_ref, m, L

    def _sector_constraints(self, t, x_ref):
        H    = self.hessian(t, x_ref)
        eigs = np.linalg.eigvals(H)
        return float(min(eigs)), float(max(eigs))


# ---------------------------------------------------------------------------
# Planar robot Jacobian helper
# ---------------------------------------------------------------------------

def _planar_jacobian(theta, link_lengths=None):
    """Geometric Jacobian for a planar nR arm."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n     = theta.size
    if link_lengths is None:
        link_lengths = np.ones(n, dtype=float)
    else:
        link_lengths = np.asarray(link_lengths, dtype=float).reshape(n)

    alpha = np.cumsum(theta)
    wsin  = link_lengths * np.sin(alpha)
    wcos  = link_lengths * np.cos(alpha)
    sx    = np.cumsum(wsin[::-1])[::-1]
    cx    = np.cumsum(wcos[::-1])[::-1]

    J = np.empty((2, n), dtype=float)
    J[0, :] = -sx
    J[1, :] =  cx
    return J


# ---------------------------------------------------------------------------
# Robotic ellipse tracking
# ---------------------------------------------------------------------------

class Robotic_Ellipse_Tracking(ObjectiveFunction):
    """Joint-velocity tracking objective for a planar robot arm following an elliptic end-effector path."""

    def __init__(self,
                 n=5,
                 theta_init=np.array([[3*np.pi/4], [-np.pi/4],
                                      [-np.pi/4],  [np.pi/6], [np.pi/3]]),
                 J_func=None,
                 mu=10.0, rho=1,
                 a=0.6, b=0.3, T=10.0,
                 center=(0.0, 0.0)):
        super().__init__(nx=n)
        self.dt = 0.05

        if theta_init.shape != (n, 1):
            raise ValueError(
                f"theta_init must have shape (n, 1) but has shape {theta_init.shape}"
            )

        self.name      = 'robotic_ellipse_tracking'
        self.current_t = 0.0
        self.theta     = np.array(theta_init, dtype=float).reshape(n, 1)
        self.theta0    = self.theta.copy()
        self.J_func    = J_func if J_func is not None else _planar_jacobian
        self.mu        = float(mu)
        self.rho       = float(rho)
        self.a, self.b = float(a), float(b)
        self.omega     = 2 * np.pi / float(T)
        self.cx, self.cy = center
        self.x_k = np.zeros((n, 1))
        self.update(0)

    def r_des(self, t):
        return np.array([[self.cx + self.a * np.cos(self.omega * t)],
                         [self.cy + self.b * np.sin(self.omega * t)]])

    def rdot_des(self, t):
        return np.array([[-self.a * self.omega * np.sin(self.omega * t)],
                         [ self.b * self.omega * np.cos(self.omega * t)]])

    def update(self, t):
        self.current_t = t * self.dt
        self.Jt   = self.J_func(self.theta)
        self.rdot = self.rdot_des(self.current_t)
        self.q    = self.mu * (self.theta - self.theta0)
        self.H    = np.eye(self.nx) + self.rho * (self.Jt.T @ self.Jt)

    def eval(self, x, prev_t=False):
        constraint = self.Jt @ x - self.rdot
        penalty    = 0.5 * self.rho * (constraint.T @ constraint)
        f = 0.5 * x.T @ x + self.q.T @ x + penalty
        return f.reshape(-1)

    def gradient(self, x):
        x    = x.reshape(-1, self.nx)
        B    = x.shape[0]
        q    = self.q
        J    = self.Jt
        rdot = self.rdot

        x_vec    = x.reshape(-1, 1)
        I_B      = np.eye(B)
        J_ext    = np.kron(I_B, J)
        q_ext    = np.tile(q,    (B, 1))
        rdot_ext = np.tile(rdot, (B, 1))

        r_vec = J_ext @ x_vec - rdot_ext
        g = x_vec + q_ext + self.rho * J_ext.T @ r_vec
        return g

    def hessian(self):
        return self.H

    def get_objective_info(self):
        rhs    = self.rho * (self.Jt.T @ self.rdot) - self.q
        x_star = np.linalg.solve(self.H, rhs)
        eigs   = np.linalg.eigvalsh(self.H)
        return x_star, float(eigs.min()), float(eigs.max())

    def step_theta(self, x_k=None):
        x = x_k if x_k is not None else self.x_k
        if x_k is not None:
            self.x_k = x_k
        self.theta = self.theta + self.dt * x

    def update_joint_angles(self, x_optimized):
        self.step_theta(x_optimized)


# ---------------------------------------------------------------------------
# QP with equality constraint (penalised)
# ---------------------------------------------------------------------------

class QP(ObjectiveFunction):
    """Time-varying quadratic objective with equality constraint enforced via quadratic penalty."""

    def __init__(self, rho=1.0):
        super().__init__(nx=2)
        self.name      = 'qp_unconstrained'
        self.current_t = 0.0
        self.rho       = float(rho)

    def P(self, t):
        return np.array([[8.0 + np.sin(t),   0.9*np.cos(t)],
                         [0.9*np.cos(t),      10.0 - 0.5*np.cos(t)]])

    def q(self, t):
        return np.array([[-2.0*np.cos(2*t)], [-2.0*np.sin(2*t)]])

    def A(self, t):
        return np.array([[2.0*np.cos(3*t), np.sin(3*t)]])

    def b(self, t):
        return np.array([np.sin(t)])

    def update(self, t):
        self.current_t = t

    def eval(self, x, prev_t=False):
        t       = self.current_t if not prev_t else self.current_t - 1
        P, q, A, b = self.P(t), self.q(t), self.A(t), self.b(t)
        x       = x.reshape(-1, self.nx)
        quad    = 0.5 * np.einsum('bi,ij,bj->b', x, P, x)
        lin     = (x @ q).ravel()
        r       = (x @ A.T - b).ravel()
        pen     = 0.5 / self.rho * (r**2)
        return (quad + lin + pen).reshape(-1)

    def gradient(self, x):
        t       = self.current_t
        P, q, A, b = self.P(t), self.q(t), self.A(t), self.b(t)
        x       = x.reshape(-1, self.nx)
        B       = x.shape[0]
        x_vec   = x.reshape(-1, 1)
        I_B     = np.eye(B)
        P_ext   = np.kron(I_B, P)
        q_ext   = np.tile(q, (B, 1))
        A_ext   = np.kron(I_B, A)
        b_ext   = np.tile(b, (B, 1))
        r       = A_ext @ x_vec - b_ext
        pen_g   = (1.0/self.rho) * (A_ext.T @ r)
        return P_ext @ x_vec + q_ext + pen_g

    def hessian(self, t):
        P, A = self.P(t), self.A(t)
        return P + (1.0/self.rho) * (A.T @ A)

    def get_objective_info(self):
        t    = self.current_t
        P, A = self.P(t), self.A(t)
        H    = P + (1.0/self.rho) * (A.T @ A)
        w    = np.linalg.eigvalsh(H)
        x_star = self.solve_star(t)
        return x_star, float(w.min()), float(w.max())

    def solve_star(self, t):
        P, q, A, b = self.P(t), self.q(t), self.A(t), self.b(t)
        KKT = np.block([[P, A.T], [A, np.zeros((1, 1))]])
        rhs = np.vstack([-q, b])
        return np.linalg.solve(KKT, rhs)[:2]


# ---------------------------------------------------------------------------
# Unconstrained time-varying QP
# ---------------------------------------------------------------------------

class QP_unconstrained(ObjectiveFunction):
    """Time-varying unconstrained quadratic objective (1/2) x^T P(t) x + q(t)^T x."""

    def __init__(self, rho=1.0):
        super().__init__(nx=2)
        self.name      = 'qp_constrained'
        self.current_t = 0.0

    def P(self, t):
        return np.array([[8.0 + np.sin(t),  0.9*np.cos(t)],
                         [0.9*np.cos(t),     10.0 - 0.5*np.cos(t)]])

    def q(self, t):
        return np.array([[-2.0*np.cos(2*t)], [-2.0*np.sin(2*t)]])

    def update(self, t):
        self.current_t = t

    def eval(self, x, prev_t=False):
        t    = self.current_t if not prev_t else self.current_t - 1
        x    = x.reshape(-1, self.nx)
        P, q = self.P(t), self.q(t)
        quad = 0.5 * np.einsum('bi,ij,bj->b', x, P, x)
        lin  = x @ q
        return (quad + lin).reshape(-1)

    def gradient(self, x):
        t    = self.current_t
        P, q = self.P(t), self.q(t)
        P_ext = np.kron(np.eye(x.shape[0]//self.nx), P)
        q_ext = np.tile(q, (x.shape[0]//self.nx, 1))
        return P_ext @ x + q_ext

    def hessian(self, t):
        return self.P(t)

    def get_objective_info(self):
        t    = self.current_t
        H    = self.P(t)
        w    = np.linalg.eigvalsh(H)
        x_star = self.solve_star(t).reshape(2, 1)
        return x_star, float(w.min()), float(w.max())

    def solve_star(self, t):
        return np.linalg.solve(self.P(t), -self.q(t)).reshape(-1, 1)
