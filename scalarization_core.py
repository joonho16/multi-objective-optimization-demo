"""시각화에서 공유하는 목적함수, 안정적인 log-sum-exp, 최적화 계산."""
import numpy as np

C1 = np.array([-2., -1.])
C2 = np.array([1., -2.])
IDEAL = np.array([-.01, -.01])  # 각 함수의 최소 0보다 epsilon=0.01 작게 설정.


def objectives(theta):
    theta = np.asarray(theta)
    return np.stack([.5*np.sum((theta-c)**2, axis=-1) for c in (C1, C2)], axis=-1)


def smooth_terms(values, weights, mu, ideal=IDEAL):
    if mu <= 0:
        raise ValueError('mu must be positive')
    a = (np.asarray(values)-ideal)*weights
    maximum = np.max(a, axis=-1, keepdims=True)
    exp = np.exp((a-maximum)/mu)
    total = exp.sum(axis=-1, keepdims=True)
    probabilities = exp/total
    value = maximum[..., 0] + mu*np.log(total[..., 0])
    return value, probabilities


def smooth_gradient(theta, weights, mu):
    _, p = smooth_terms(objectives(theta), weights, mu)
    return p[0]*weights[0]*(theta-C1) + p[1]*weights[1]*(theta-C2)


def smooth_solution(weights, mu):
    """이 볼록 이차함수 예제의 해: 중심들을 잇는 선분에서 도함수의 근을 찾는다."""
    if weights[0] == 0:
        return C2.copy()
    if weights[1] == 0:
        return C1.copy()
    lo, hi = 0., 1.
    for _ in range(65):
        t = (lo+hi)/2
        theta = C1+t*(C2-C1)
        derivative = smooth_gradient(theta, weights, mu) @ (C2-C1)
        if derivative > 0:
            hi = t
        else:
            lo = t
    return C1+(lo+hi)/2*(C2-C1)


def mgda_direction(theta):
    g1, g2 = theta-C1, theta-C2
    delta = g1-g2
    alpha = float(np.clip(-(g2@delta)/(delta@delta), 0, 1))
    return g2+alpha*delta


def update(theta, method, weights, mu, eta):
    """하강 방향을 구하고 Armijo backtracking으로 실제 감소를 확인한다."""
    if method == 'MGDA':
        direction = mgda_direction(theta)
        score = objectives
    elif method == 'LS':
        direction = theta-(weights[0]*C1+weights[1]*C2)
        score = lambda point: objectives(point)@weights
    elif method == 'STCH':
        direction = smooth_gradient(theta, weights, mu)
        score = lambda point: smooth_terms(objectives(point), weights, mu)[0]
    else:
        raise ValueError(method)
    norm2 = float(direction@direction)
    if norm2 < 1e-18:
        return theta.copy(), 0.
    old = score(theta)
    for _ in range(60):
        candidate = theta-eta*direction
        if np.all(score(candidate) <= old-1e-4*eta*norm2):
            return candidate, eta
        eta *= .5
    return theta.copy(), 0.


def nonconvex_objectives(t):
    t = np.asarray(t)
    return np.stack([t, 1-t*t], axis=-1)


def nonconvex_tch_solution(weights):
    """max(증가하는 항, 감소하는 항)의 정확한 최저점: 교점 또는 끝점."""
    def gap(t):
        a = weights*(nonconvex_objectives(t)-IDEAL)
        return a[0]-a[1]
    if gap(0) >= 0:
        return 0.
    if gap(1) <= 0:
        return 1.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        if gap(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def nonconvex_solutions(weights, mu):
    if abs(weights[0]-weights[1]) < 1e-12:
        ls = np.array([0., 1.])  # 같은 가중치에서는 두 끝점이 모두 전역해.
    else:
        ls = np.array([0. if weights[0] > weights[1] else 1.])
    grid = np.linspace(0, 1, 10001)
    values, _ = smooth_terms(nonconvex_objectives(grid), weights, mu)
    return ls, nonconvex_tch_solution(weights), float(grid[np.argmin(values)])
