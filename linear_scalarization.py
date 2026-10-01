"""실행: python linear_scalarization.py

X=R², m=2인 예제. 선호도 λ를 고정하고 g(θ|λ)=Σ λ_i f_i(θ)를 최소화한다.
MGDA.py와 같은 목적함수를 사용한다. 이 이차함수 예제는 정확한 해를 구할 수 있다.
일반적인 비볼록 문제에서는 가중합만으로 Pareto front 전체를 얻지 못할 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Button, Slider


C1 = np.array([-2., -1.])
C2 = np.array([1., -2.])
BLUE, PURPLE, GREEN, ORANGE = '#2563eb', '#9333ea', '#15803d', '#ea580c'


def objectives(theta):
    theta = np.asarray(theta)
    return np.stack([.5 * np.sum((theta - c)**2, axis=-1)
                     for c in (C1, C2)], axis=-1)


def scalarization(theta, weights):
    return objectives(theta) @ weights


def solve(weights):
    # ∇g = λ1(θ−C1) + λ2(θ−C2) = θ−(λ1 C1+λ2 C2) = 0.
    # Hessian이 I이므로 이 정지점이 유일한 전역 최솟값을 주는 입력이다.
    return weights[0] * C1 + weights[1] * C2


def vec(v):
    return f'({v[0]:.3f}, {v[1]:.3f})'


def main():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'AppleGothic'):
        if name in fonts:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9,
                         'axes.titlesize': 11, 'axes.titlepad': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(16, 10), facecolor='#f5f7fb')
    grid = fig.add_gridspec(2, 2, left=.055, right=.665, bottom=.275, top=.84,
                           wspace=.34, hspace=.52)
    ax = fig.add_subplot(grid[0, 0])
    weighted_ax = fig.add_subplot(grid[0, 1])
    objective_ax = fig.add_subplot(grid[1, 0])
    preference_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle('Linear scalarization · 선호도를 정하고, 가중합을 최소화', fontsize=19, y=.978)
    fig.text(.5, .925,
             r'$\min_{\theta\in\mathbb{R}^2}\ g(\theta\mid\lambda)'
             r'=\lambda_1 f_1(\theta)+\lambda_2 f_2(\theta),'
             r'\quad \lambda_1+\lambda_2=1,\quad\lambda_i\geq 0$',
             fontsize=16, ha='center')
    fig.text(.5, .883,
             'f1 = ½[(x+2)²+(y+1)²]     f2 = ½[(x−1)²+(y+2)²]     '
             'θ=(x,y)는 입력, λ는 사용자가 정하는 선호도', ha='center', color='#475569')
    explanation = fig.text(.715, .845, '', va='top', fontsize=10, linespacing=1.7)
    calculation = fig.text(.715, .615, '', va='top', fontsize=10, linespacing=1.7)
    result = fig.text(.715, .36, '', va='top', fontsize=10, linespacing=1.7)
    status = fig.text(.055, .025, 'λ1을 바꾸고 「최적해로 이동」을 눌러 보세요. 입력 지도 클릭으로 출발점도 바꿀 수 있습니다.',
                      color='#475569', fontsize=10)
    weight_slider = Slider(fig.add_axes([.14, .155, .38, .025]),
                           '선호도 λ1', 0., 1., valinit=.5, valfmt='%.2f', color=BLUE)
    eta_slider = Slider(fig.add_axes([.14, .09, .38, .025]),
                        'GD 이동 크기 η', .01, 1., valinit=.2, valfmt='%.2f')
    fig.text(.055, .213, 'λ2 = 1−λ1 자동 적용 · λ1이 클수록 f1에 더 큰 가중치', fontsize=11)
    state = {'theta': np.zeros(2), 'path': [np.zeros(2)], 'solutions': []}
    buttons = []

    def render():
        w = np.array([weight_slider.val, 1 - weight_slider.val])
        theta = state['theta']
        optimum = solve(w)
        current_values, optimal_values = objectives(theta), objectives(optimum)
        g = scalarization(theta, w)
        best_g = scalarization(optimum, w)
        grad = theta - optimum
        path = np.array(state['path'])
        for panel in (ax, weighted_ax, objective_ax, preference_ax):
            panel.clear()
            panel.set_facecolor('white')
            panel.grid(alpha=.15)
            panel.tick_params(labelsize=8)

        xx, yy = np.meshgrid(np.linspace(-4, 3, 100), np.linspace(-4, 2, 100))
        positions = np.stack([xx, yy], axis=-1)
        values = objectives(positions)
        for i, center, color in ((0, C1, BLUE), (1, C2, PURPLE)):
            ax.contour(xx, yy, values[..., i], levels=[.25, 1, 2.5, 5, 10, 20],
                       colors=color, alpha=.45, linewidths=.8)
            ax.scatter(*center, color=color, marker='*', s=110,
                       label=f'f{i+1}가 최소인 입력')
        ax.plot([C1[0], C2[0]], [C1[1], C2[1]], '--', color=GREEN, lw=1)
        ax.scatter(*theta, color='black', s=35, zorder=6, label='현재 θ')
        ax.scatter(*optimum, color=ORANGE, marker='*', s=140, zorder=7, label='선택한 λ의 최적해 θ*')
        ax.set(title='① 입력 공간 · 두 목적함수의 등고선', xlabel='입력 x', ylabel='입력 y',
               xlim=(-4, 3), ylim=(-4, 2))
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=7, loc='upper right')

        levels = best_g + np.array([.1, .4, 1, 2, 4, 8, 16])
        contours = weighted_ax.contour(xx, yy, values @ w, levels=levels,
                                       cmap='viridis', linewidths=1)
        weighted_ax.clabel(contours, fmt='%.2f', fontsize=7)
        weighted_ax.plot(path[:, 0], path[:, 1], '.-', color='#64748b', lw=1)
        weighted_ax.scatter(*theta, color='black', s=35, zorder=6)
        weighted_ax.scatter(*optimum, color=ORANGE, marker='*', s=140, zorder=7)
        weighted_ax.annotate('', theta - eta_slider.val * grad, theta,
                             arrowprops=dict(arrowstyle='->', lw=2, color=GREEN))
        weighted_ax.set(title=f'② 가중합 g = {w[0]:.2f} f1 + {w[1]:.2f} f2',
                        xlabel='입력 x', ylabel='입력 y', xlim=(-4, 3), ylim=(-4, 2))
        weighted_ax.set_aspect('equal', adjustable='box')

        preferences = np.linspace(0, 1, 201)
        optimal_inputs = preferences[:, None]*C1 + (1-preferences[:, None])*C2
        frontier = objectives(optimal_inputs)
        path_values = objectives(path)
        limit = max(5., path_values.max(), current_values.max()) * 1.12
        u = np.linspace(0, limit, 250)
        distance = np.linalg.norm(C1-C2)
        objective_ax.fill_between(u, .5*(np.sqrt(2*u)-distance)**2,
                                  .5*(np.sqrt(2*u)+distance)**2,
                                  color='#e2e8f0', alpha=.65)
        objective_ax.plot(frontier[:, 0], frontier[:, 1], color=GREEN, lw=2,
                          label='Pareto front (이 볼록 예제)')
        # 가중합이 같은 점들의 직선. 최저 등고선은 최적해에서 front에 닿는다.
        for level, color, style, label in ((g, '#64748b', ':', '현재 가중합 등고선'),
                                           (best_g, ORANGE, '--', '최소 가중합 등고선')):
            if w[1] > 1e-10:
                objective_ax.plot(u, (level-w[0]*u)/w[1], style, color=color, lw=1, label=label)
            else:
                objective_ax.axvline(level/w[0], color=color, linestyle=style, label=label)
        objective_ax.plot(path_values[:, 0], path_values[:, 1], '.-', color='#64748b', lw=1)
        objective_ax.scatter(*current_values, color='black', s=35, zorder=6)
        objective_ax.scatter(*optimal_values, color=ORANGE, marker='*', s=140, zorder=7)
        if state['solutions']:
            solutions = np.array(state['solutions'])
            objective_ax.scatter(solutions[:, 1], solutions[:, 2], c=solutions[:, 0],
                                 cmap='cool', vmin=0, vmax=1, s=40, edgecolors='white',
                                 zorder=5, label=f'풀어서 모은 해 {len(solutions)}개')
        objective_ax.set(title='③ 목적함수 공간 · 직선을 왼쪽 아래로',
                         xlabel='f1(θ)', ylabel='f2(θ)', xlim=(0, limit), ylim=(0, limit))
        objective_ax.legend(fontsize=6.5, loc='upper right')

        preference_ax.plot(preferences, frontier[:, 0], color=BLUE, label='최적해에서 f1')
        preference_ax.plot(preferences, frontier[:, 1], color=PURPLE, label='최적해에서 f2')
        preference_ax.axvline(w[0], color=ORANGE, linestyle='--', lw=1)
        preference_ax.scatter([w[0], w[0]], optimal_values, c=[BLUE, PURPLE], s=45, zorder=5)
        preference_ax.set(title='④ λ를 바꾸어 얻는 최적해들의 함수값', xlabel='λ1 (λ2 = 1−λ1)',
                          ylabel='각 λ의 최적해 θ*에서 평가', xlim=(0, 1), ylim=(-.2, 5.4))
        preference_ax.legend(fontsize=8)

        explanation.set_text('1. 선호도는 simplex 위의 점\n'
            '여기서는 m=2, λ1+λ2=1, λi≥0\n'
            f'λ = ({w[0]:.2f}, {w[1]:.2f})\n'
            'λ=(1,0): f1만 최소화\n'
            'λ=(0,1): f2만 최소화\n'
            'λ=(0.5,0.5): 같은 가중치\n'
            '각 최적화 중에는 λ를 고정합니다.')
        calculation.set_text('2. 가중합을 미분해 최적해 구하기\n'
            '∇g = λ1(θ−c1) + λ2(θ−c2)\n'
            '     = θ − (λ1 c1 + λ2 c2)\n'
            '∇g=0 → θ* = λ1 c1 + λ2 c2\n'
            f'θ* = (1−3λ1, −2+λ1) = {vec(optimum)}\n'
            '이 예제는 Hessian=I이므로 전역 최소.\n'
            f'f(θ*) = {vec(optimal_values)}\n'
            f'g(θ*|λ) = 5λ1λ2 = {best_g:.4f}')
        result.set_text('3. 현재 위치에서의 계산\n'
            f'θ = {vec(theta)}\n'
            f'f(θ) = {vec(current_values)}\n'
            f'g = {w[0]:.2f}×{current_values[0]:.3f}'
            f' + {w[1]:.2f}×{current_values[1]:.3f}\n'
            f'   = {g:.4f},  ∇g = {vec(grad)}\n'
            'GD 한 걸음: θ ← θ − η∇g')
        fig.canvas.draw_idle()

    def weights():
        return np.array([weight_slider.val, 1-weight_slider.val])

    def change_weight(_value):
        state['path'] = [state['theta'].copy()]
        status.set_text('λ가 바뀌었습니다. 주황 별은 새 선호도의 최적해, 검정 점은 현재 입력입니다.')
        render()

    def exact_solve(_event):
        w = weights()
        state['theta'] = solve(w)
        state['path'].append(state['theta'].copy())
        state['solutions'] = [row for row in state['solutions'] if abs(row[0]-w[0]) > 1e-9]
        state['solutions'].append((w[0], *objectives(state['theta'])))
        status.set_text(f'λ={vec(w)}를 고정하고 최소화 완료: θ*={vec(state["theta"])}. 주황 별과 검정 점이 일치합니다.')
        render()

    def step(_event):
        grad = state['theta'] - solve(weights())
        state['theta'] = state['theta'] - eta_slider.val * grad
        state['path'].append(state['theta'].copy())
        status.set_text('고정한 λ의 가중합을 줄이는 GD 한 걸음. 각 f1, f2가 모두 감소해야 하는 것은 아닙니다.')
        render()

    def sweep(_event):
        state['solutions'] = [(a, *objectives(solve(np.array([a, 1-a]))))
                              for a in np.linspace(0, 1, 11)]
        status.set_text('λ1=0, 0.1, …, 1의 11가지 선호도로 각각 최소화했습니다. ③의 색 점들이 서로 다른 해입니다.')
        render()

    def reset(_event):
        state.update(theta=np.zeros(2), path=[np.zeros(2)], solutions=[])
        weight_slider.eventson = eta_slider.eventson = False
        weight_slider.reset()
        eta_slider.reset()
        weight_slider.eventson = eta_slider.eventson = True
        status.set_text('초기화했습니다. λ1을 바꾸고 최적해를 비교하세요.')
        render()

    def click(event):
        if event.inaxes is ax and event.button == 1 and event.xdata is not None:
            if getattr(getattr(fig.canvas, 'toolbar', None), 'mode', ''):
                return
            state['theta'] = np.array([event.xdata, event.ydata])
            state['path'] = [state['theta'].copy()]
            status.set_text('출발점을 바꿨습니다. λ를 고정한 채 GD 한 걸음 또는 최적해로 이동을 누르세요.')
            render()

    for label, callback, x, y in (
        ('최적해로 이동', exact_solve, .61, .145), ('GD 한 걸음', step, .80, .145),
        ('11가지 선호도로 풀기', sweep, .61, .08), ('처음으로', reset, .80, .08)):
        button = Button(fig.add_axes([x, y, .16, .045]), label)
        button.on_clicked(callback)
        buttons.append(button)
    weight_slider.on_changed(change_weight)
    eta_slider.on_changed(lambda value: render())
    fig.canvas.mpl_connect('button_press_event', click)
    render()
    plt.show()


if __name__ == '__main__':
    main()
