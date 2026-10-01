"""실행: python MGDA.py"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RangeSlider, Slider

# Ubuntu의 한글 글꼴을 우선 사용하고, 다른 운영체제에서는 설치된 한글 글꼴을 찾는다.
_fonts = {font.name for font in font_manager.fontManager.ttflist}
for _name in ('Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'AppleGothic'):
    if _name in _fonts:
        plt.rcParams['font.family'] = _name
        break
plt.rcParams['axes.unicode_minus'] = False

# f1, f2는 모두 작을수록 좋다. (0, 0)에서 기존 예제와 같은 gradient가 나온다.
# f1=1/2*((x+2)^2+(y+1)^2), f2=1/2*((x-1)^2+(y+2)^2)
c1, c2 = np.array([-2., -1.]), np.array([1., -2.])
theta = np.zeros(2)
eta = .2
history = [theta.copy()]


def objectives(p):
    p = np.asarray(p)
    return np.stack([.5 * np.sum((p - c) ** 2, axis=-1) for c in (c1, c2)], axis=-1)


def vec(p):
    return f'({p[0]:.3f}, {p[1]:.3f})'


def objective_gradients(p):
    return p - c1, p - c2


def solve_qp(g1, g2):
    """정확히 풀이: min_{0<=a<=1} ||g2+a(g1-g2)||^2."""
    delta = g1 - g2
    denom = float(delta @ delta)
    raw = -float(g2 @ delta) / denom if denom else .5
    alpha = float(np.clip(raw, 0, 1))
    return alpha, g2 + alpha * delta, raw


g1, g2 = objective_gradients(theta)
alpha_star, d_star, raw_alpha = solve_qp(g1, g2)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.titlepad': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(16, 10), facecolor='#f5f7fb')
grid = fig.add_gridspec(2, 2, left=.055, right=.665, bottom=.285, top=.84,
                       wspace=.32, hspace=.53)
ax = fig.add_subplot(grid[0, 0])
fx = fig.add_subplot(grid[0, 1])
vx = fig.add_subplot(grid[1, 0])
qx = fig.add_subplot(grid[1, 1])
fig.suptitle('MGDA 공부하기: 입력과 결과를 함께 보기', fontsize=20, y=.977)
fig.text(.5, .925, r'$f_1(x,y)=\frac{1}{2}[(x+2)^2+(y+1)^2]$'
         '       '
         r'$f_2(x,y)=\frac{1}{2}[(x-1)^2+(y+2)^2]$',
         ha='center', fontsize=16)
fig.text(.5, .88, '왼쪽 위: 어떤 x, y를 선택했나?     오른쪽 위: 그 선택의 f1, f2는 얼마인가?', ha='center', color='#475569')
info = fig.text(.715, .845, '', va='top', fontsize=10, linespacing=1.65)
qp_info = fig.text(.715, .635, '', va='top', fontsize=10, linespacing=1.65)
step_info = fig.text(.715, .395, '', va='top', fontsize=10, linespacing=1.65)
def render():
    global g1, g2, d_star, alpha_star, raw_alpha
    g1, g2 = objective_gradients(theta)
    alpha_star, d_star, raw_alpha = solve_qp(g1, g2)
    alpha = alpha_slider.val
    d = g2 + alpha * (g1 - g2)
    f = objectives(theta)
    future = theta - eta_slider.val * d
    fn = objectives(future)
    for a in (ax, fx, vx, qx):
        a.clear()
        a.set_facecolor('white')
        a.grid(alpha=.15)
        a.tick_params(labelsize=8)

    # ① 실제 함수 f1, f2의 등고선과 현재 위치 θ.
    points = np.vstack([history, future])
    lo, hi = min(-4., points.min()-1), max(3., points.max()+1)
    X, Y = np.meshgrid(np.linspace(lo, hi, 90), np.linspace(lo, hi, 90))
    for c, color, label in ((c1, 'tab:blue', 'f1'), (c2, 'tab:purple', 'f2')):
        Z = .5*((X-c[0])**2+(Y-c[1])**2)
        ax.contour(X, Y, Z, levels=[.5, 2, 5, 10, 20], colors=color, alpha=.55)
        ax.scatter(*c, marker='*', color=color, s=100, label=f'{label}가 최소인 입력')
    path = np.array(history)
    ax.plot(path[:, 0], path[:, 1], '.-', color='#64748b', lw=1, alpha=.7)
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], '--', color='tab:green', lw=1.3)
    ax.scatter(*theta, color='black', s=40, zorder=6, label='현재 입력 θ')
    ax.annotate('', theta-eta_slider.val*d, theta,
                arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2.5))
    ax.scatter(*future, color='tab:orange', s=25, zorder=5)
    ax.set(xlabel='입력 x', ylabel='입력 y', title='① 입력 공간 · 클릭해서 θ 변경',
           xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    # 같은 입력 θ를 결과 (f1(θ), f2(θ))로 변환한 목적함수 공간.
    values = objectives(path)
    limit = max(5., f.max(), fn.max(), values.max()) * 1.12
    u = np.linspace(0, limit, 240)
    distance = np.linalg.norm(c2 - c1)
    # 두 중심까지 거리의 삼각부등식으로 얻은 실제 도달 가능 영역.
    lower = .5 * (np.sqrt(2*u) - distance)**2
    upper = .5 * (np.sqrt(2*u) + distance)**2
    fx.fill_between(u, lower, upper, color='#e2e8f0', alpha=.65,
                    label='가능한 함수값 영역')
    fx.fill_between([0, f[0]], 0, f[1], color='tab:green', alpha=.08)
    t = np.linspace(0, 1, 160)
    pareto = objectives(c1 + t[:, None]*(c2-c1))
    fx.plot(pareto[:, 0], pareto[:, 1], color='tab:green', lw=2,
            label='Pareto front')
    fx.plot(values[:, 0], values[:, 1], '.-', color='#64748b', lw=1, alpha=.7)
    fx.scatter(*f, color='black', s=40, zorder=6, label='현재 (f1, f2)')
    fx.annotate('', fn, f, arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2))
    fx.scatter(*fn, color='tab:orange', s=35, zorder=5, label='현재 α로 이동 시')
    fx.text(.03, .03, '왼쪽 아래로 갈수록 두 값 모두 감소',
            transform=fx.transAxes, fontsize=8, color='tab:green')
    fx.set(xlabel='f1(θ) — 작을수록 좋음', ylabel='f2(θ) — 작을수록 좋음',
           title='② 목적함수 공간 · 동일한 θ의 결과', xlim=(0, limit), ylim=(0, limit))
    fx.legend(fontsize=7, loc='upper right')

    # ② gradient들을 잇는 선분에서 원점에 가장 가까운 점을 찾는다.
    vx.axhline(0,color='.7'); vx.axvline(0,color='.7')
    vx.plot([g1[0],g2[0]],[g1[1],g2[1]], color='.7', lw=3)
    for v, color, label in ((g1,'tab:blue','∇f1 증가'),(g2,'tab:purple','∇f2 증가'),
                             (d,'tab:orange',f'd(α), α={alpha:.2f}'),
                             (d_star,'tab:green',f'd*, α*={alpha_star:.3f}')):
        vx.annotate('',v,(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=2))
        vx.annotate(label, v, color=color, fontsize=8)
    extent = max(1., np.abs(np.vstack([g1, g2, d_star])).max()) * 1.45
    vx.set(xlabel='gradient x 성분',ylabel='gradient y 성분',title='③ gradient 조합',
           xlim=(-extent, extent), ylim=(-extent, extent))
    vx.set_aspect('equal', adjustable='box')

    # ③ QP를 직접 그려서 최솟값을 확인한다.
    aa=np.linspace(0,1,101); delta=g1-g2
    q= np.array([np.linalg.norm(g2+x*delta)**2 for x in aa])
    qx.plot(aa,q,color='.3')
    qx.axvspan(*bounds.val, color='tab:orange', alpha=.07)
    qx.scatter([alpha],[np.dot(d,d)],color='tab:orange',label='현재 α')
    qx.scatter([alpha_star],[np.dot(d_star,d_star)],color='tab:green',marker='*',s=150,
               label=f'최적 α*={alpha_star:.3f}')
    qx.set(xlabel='α (0 ≤ α ≤ 1)',ylabel='q(α)=||d(α)||²',title='④ QP · 곡선의 최저점')
    qx.legend(fontsize=8)

    A=float(delta@delta); B=float(2*g2@delta); C=float(g2@g2)
    info.set_text('입력 → 함수값 → 미분\n'
        f'θ = {vec(theta)}\n'
        f'(f1, f2) = {vec(f)}\n'
        f'g1 = (x+2, y+1) = {vec(g1)}\n'
        f'g2 = (x−1, y+2) = {vec(g2)}\n'
        'gradient는 증가 방향, −d는 이동 방향.\n'
        '검정: 현재 위치 / 주황: 이동 미리보기')
    qp_info.set_text('QP 계산 · θ를 고정하고 α를 구함\n'
        'd(α) = α g1 + (1−α) g2\n'
        'min ||d(α)||²,  0 ≤ α ≤ 1\n'
        f'q = {A:.3f}α² {B:+.3f}α {C:+.3f}\n'
        f'q′ = {2*A:.3f}α {B:+.3f} = 0\n'
        f'α = {raw_alpha:.3f} → [0,1]로 제한\n'
        f'α* = {alpha_star:.3f},  d* = {vec(d_star)}\n'
        f'||d*||² = {d_star@d_star:.4f}')
    verdict = ('두 함수 모두 감소' if np.all(fn < f) else '현재 α로는 둘 다 감소하지 않음')
    if np.linalg.norm(d_star) < 1e-8:
        verdict = 'd*=0: Pareto 정지점'
    step_info.set_text(f'이동 미리보기 · α={alpha:.3f}, η={eta_slider.val:.2f}\n'
        'θ_new = θ − η d\n'
        f'f1: {f[0]:.4f} → {fn[0]:.4f}\n'
        f'f2: {f[1]:.4f} → {fn[1]:.4f}\n'
        f'방향미분: ({-g1@d:+.3f}, {-g2@d:+.3f})\n'
        + verdict)
    fig.canvas.draw_idle()

status = fig.text(.055, .025, '입력 지도를 클릭하거나 α를 조절하세요. MGDA 한 걸음은 QP의 최적 α*를 사용합니다.', fontsize=10, color='#475569')
fig.text(.055, .203, '실험 조작', fontsize=12, weight='bold')

bounds = RangeSlider(fig.add_axes([0.135, 0.145, 0.255, 0.024]),
                     'α 탐색 범위', 0.0, 1.0, valinit=(0.0, 1.0), valfmt='%.2f')
alpha_slider = Slider(fig.add_axes([0.135, 0.088, 0.255, 0.024]),
                      '혼합 비율 α', 0.0, 1.0, valinit=0.5, valfmt='%.3f',
                      color='tab:orange')
speed = Slider(fig.add_axes([0.515, 0.088, 0.20, 0.024]),
               'α 재생 속도', 0.05, 1.0, valinit=0.25, valfmt='%.2f')
play = Button(fig.add_axes([0.78, 0.135, 0.09, 0.041]), 'α 재생')
reset = Button(fig.add_axes([0.885, 0.135, 0.09, 0.041]), '처음으로')
state = {'playing': False, 'direction': 1, 'last_time': time.monotonic()}
eta_slider = Slider(fig.add_axes([0.515, 0.145, 0.20, 0.024]),
                    '이동 크기 η', 0.01, 1.0, valinit=eta)


def set_alpha(value):
    value = float(np.clip(value, *bounds.val))
    # Programmatic updates must not trigger the manual slider callback.
    alpha_slider.eventson = False
    try:
        alpha_slider.set_val(value)
    finally:
        alpha_slider.eventson = True
    render()


def set_playing(playing):
    state['playing'] = playing
    state['last_time'] = time.monotonic()
    play.label.set_text('일시정지' if playing else 'α 재생')
    fig.canvas.draw_idle()


def change_alpha(value):
    set_playing(False)
    set_alpha(value)


def change_range(values):
    set_alpha(alpha_slider.val)
    state['last_time'] = time.monotonic()


def reset_animation(_event):
    set_playing(False)
    bounds.reset()
    speed.reset()
    state['direction'] = 1
    theta[:] = 0
    history[:] = [theta.copy()]
    eta_slider.reset()
    set_alpha(0.5)
    status.set_text('초기 입력 (0, 0) → 함수값 (2.5, 2.5). α*=0.5이고 η=0.2이면 두 함수값은 2.05가 됩니다.')


def solve_button(_event):
    set_playing(False)
    bounds.eventson = False
    bounds.set_val((0, 1))
    bounds.eventson = True
    render()
    alpha_slider.eventson = False
    alpha_slider.set_val(alpha_star)
    alpha_slider.eventson = True
    status.set_text(f'QP의 전체 범위 해: α*={alpha_star:.4f}, d*={vec(d_star)}, ||d*||²={d_star@d_star:.4f}')
    render()


def take_step(_event):
    set_playing(False)
    solve_button(None)
    if np.linalg.norm(d_star) < 1e-8:
        status.set_text('Pareto 정지점: d*=0이므로 MGDA는 멈춥니다. 목적함수 공간에서 초록 곡선 위의 점입니다.')
        fig.canvas.draw_idle()
        return
    theta[:] -= eta_slider.val * d_star
    history.append(theta.copy())
    status.set_text(f'MGDA 한 걸음 완료: 새 입력 θ={vec(theta)} → 새 함수값 (f1, f2)={vec(objectives(theta))}. 회색 선은 이동 기록입니다.')
    render()


def click_map(event):
    if event.inaxes is ax and event.button == 1 and event.xdata is not None:
        if getattr(getattr(fig.canvas, 'toolbar', None), 'mode', ''):
            return
        theta[:] = (event.xdata, event.ydata)
        history[:] = [theta.copy()]
        set_playing(False)
        status.set_text('지도를 눌러 위치를 옮겼습니다. gradient와 QP 해를 확인하세요.')
        render()


solve_button_widget = Button(fig.add_axes([0.78, 0.078, 0.09, 0.041]), 'QP 풀기')
solve_button_widget.on_clicked(solve_button)
step_button = Button(fig.add_axes([0.885, 0.078, 0.09, 0.041]), 'MGDA 한 걸음')
step_button.on_clicked(take_step)
fig.canvas.mpl_connect('button_press_event', click_map)
eta_slider.on_changed(lambda value: render())


def animate(_frame):
    now = time.monotonic()
    elapsed = now - state['last_time']
    state['last_time'] = now
    lower, upper = bounds.val
    width = upper - lower
    if not state['playing'] or width <= 0:
        return ()
    offset = alpha_slider.val - lower
    phase = offset if state['direction'] > 0 else 2 * width - offset
    phase = (phase + elapsed * speed.val) % (2 * width)
    state['direction'] = 1 if phase < width else -1
    set_alpha(lower + (phase if phase < width else 2 * width - phase))
    return ()


bounds.on_changed(change_range)
alpha_slider.on_changed(change_alpha)
play.on_clicked(lambda event: set_playing(not state['playing']))
reset.on_clicked(reset_animation)
change_range(bounds.val)
# Keep this reference alive so Matplotlib does not discard the animation.
animation = FuncAnimation(fig, animate, interval=150, cache_frame_data=False)

if __name__ == '__main__':
    plt.show()
