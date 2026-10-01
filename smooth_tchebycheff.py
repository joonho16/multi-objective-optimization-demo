"""python smooth_tchebycheff.py: 기존 이차함수 예제로 Smooth Tchebycheff 공부하기."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Button, Slider
from scalarization_core import (C1, C2, IDEAL, objectives, smooth_terms,
                                smooth_gradient, smooth_solution, update)


COLORS = {'MGDA': '#15803d', 'LS': '#2563eb', 'STCH': '#ea580c'}


def style():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'AppleGothic'):
        if name in fonts:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9,
                         'axes.titlesize': 11, 'axes.titlepad': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})


def vec(value):
    return f'({value[0]:.3f}, {value[1]:.3f})'


def main(compare=False):
    style()
    methods = ('MGDA', 'LS', 'STCH') if compare else ('STCH',)
    fig = plt.figure(figsize=(16, 10), facecolor='#f5f7fb')
    grid = fig.add_gridspec(2, 2, left=.055, right=.66, top=.835, bottom=.285,
                           wspace=.34, hspace=.52)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    title = '같은 출발점에서 MGDA · 선형 LS · Smooth Tchebycheff 비교' if compare else 'Smooth Tchebycheff · max를 부드럽게 만들기'
    fig.suptitle(title, fontsize=19, y=.975)
    fig.text(.5, .91,
             r'$g_\mu(\theta\mid\lambda)=\mu\log\sum_i\exp'
             r'\left(\lambda_i(f_i(\theta)-z_i^*)/\mu\right)$', ha='center', fontsize=17)
    fig.text(.5, .882, '기존 예제 그대로: f1=½[(x+2)²+(y+1)²], f2=½[(x−1)²+(y+2)²]   |   z*=(-0.01, -0.01)', ha='center')
    notes = fig.text(.71, .845, '', va='top', fontsize=10, linespacing=1.6)
    status = fig.text(.055, .025, '①을 클릭해 출발점 변경 → 선호도와 μ 조절 → 한 걸음 또는 100걸음.', color='#475569')
    lam = Slider(fig.add_axes([.13, .17, .38, .022]), '선호도 λ1', 0, 1,
                 valinit=.8 if compare else .5, valfmt='%.2f')
    mu = Slider(fig.add_axes([.13, .115, .38, .022]), '부드러움 μ', .005, 1., valinit=.1, valfmt='%.3f')
    eta = Slider(fig.add_axes([.13, .06, .38, .022]), '시작 이동폭 η', .01, 1., valinit=.2, valfmt='%.2f')
    state = {'start': np.zeros(2), 'paths': {m: [np.zeros(2)] for m in methods}}
    buttons = []

    def weights():
        return np.array([lam.val, 1-lam.val])

    def render():
        w = weights()
        optimum = smooth_solution(w, mu.val)
        for ax in axes:
            ax.clear()
            ax.set_facecolor('white')
            ax.grid(alpha=.15)
        x, y = np.meshgrid(np.linspace(-4, 3, 90), np.linspace(-4, 2, 90))
        mesh = np.stack([x, y], axis=-1)
        mesh_values = objectives(mesh)
        minimum, _ = smooth_terms(objectives(optimum), w, mu.val)
        for i, center, color in ((0, C1, '#2563eb'), (1, C2, '#9333ea')):
            axes[0].contour(x, y, mesh_values[..., i], levels=[.25, 1, 2.5, 5, 10, 20],
                            colors=color, linewidths=.8, alpha=.55)
            axes[0].scatter(*center, color=color, marker='*', s=110, zorder=6,
                            label=f'f{i+1}가 최소인 입력')
        axes[0].plot([C1[0], C2[0]], [C1[1], C2[1]], '--', color='#15803d',
                     lw=2, label='Pareto 최적 입력들의 선분')
        axes[0].scatter(*optimum, color=COLORS['STCH'], marker='*', s=120, zorder=7,
                        label='현재 λ가 선택한 STCH 해')
        axes[0].set(title='① 입력 공간 · 파랑 f1 / 보라 f2 등고선', xlabel='입력 x', ylabel='입력 y',
                    xlim=(-4,3), ylim=(-4,2))
        axes[0].set_aspect('equal', adjustable='box')
        t = np.linspace(0, 1, 400)
        segment = C1+t[:,None]*(C2-C1)
        front = objectives(segment)
        axes[1].plot(front[:,0], front[:,1], color='#15803d', lw=2.3, label='Pareto front 전체')
        for i, center, color in ((0, C1, '#2563eb'), (1, C2, '#9333ea')):
            axes[1].scatter(*objectives(center), color=color, marker='*', s=100, zorder=6,
                            label=f'f{i+1} 최소일 때의 (f1, f2)')
        axes[1].scatter(*objectives(optimum), color=COLORS['STCH'], marker='*', s=140,
                        zorder=7, label='현재 λ가 선택한 STCH 해')
        limit = 5.6
        for method in methods:
            path = np.array(state['paths'][method])
            values = objectives(path)
            color = COLORS[method]
            axes[0].plot(path[:,0],path[:,1],'.-',color=color,label=method,ms=3,lw=1)
            axes[0].scatter(*path[-1],color=color,s=35,zorder=5)
            axes[1].plot(values[:,0],values[:,1],'.-',color=color,label=method,ms=3,lw=1)
            axes[1].scatter(*values[-1],color=color,s=35,zorder=5)
            limit = max(limit, values.max()*1.1)
            if compare:
                for index, ax in enumerate(axes[2:]):
                    ax.plot(values[:,index],color=color,label=method,lw=1.7)
                    ax.set(title=f'③ f1 변화' if index == 0 else '④ f2 변화',
                           xlabel='업데이트 횟수',ylabel=f'f{index+1}')
                    ax.legend(fontsize=8)
        axes[0].legend(fontsize=6.5, loc='upper right')
        axes[1].set(title='② 목적함수 공간 · 두 목표와 Pareto front', xlabel='f1 (작을수록 좋음)', ylabel='f2 (작을수록 좋음)',
                    xlim=(0,limit), ylim=(0,limit))
        axes[1].legend(fontsize=6.5, loc='upper right')
        current = state['paths']['STCH'][-1]
        f = objectives(current)
        g, p = smooth_terms(f,w,mu.val)
        a = w*(f-IDEAL)
        grad = smooth_gradient(current,w,mu.val)
        if not compare:
            section, _ = smooth_terms(front,w,mu.val)
            classic = np.max(w*(front-IDEAL),axis=1)
            axes[2].plot(t,classic,'--',color='#64748b',label='TCH: max(a1, a2)')
            axes[2].plot(t,section,color=COLORS['STCH'],label=f'STCH: μ={mu.val:.3f}')
            optimal_t = (optimum-C1)@(C2-C1)/10
            axes[2].scatter(optimal_t,minimum,marker='*',s=100,color=COLORS['STCH'])
            axes[2].set(title='③ 선분 단면 · 꺾인 max와 부드러운 곡선',
                        xlabel='t: θ=c1+t(c2−c1), 원래 문제는 θ∈R²',ylabel='scalarization 값')
            axes[2].legend(fontsize=8)
            axes[3].bar([0,1],p,width=.35,color=['#2563eb','#9333ea'],label='softmax p')
            axes[3].bar([.38,1.38],p*w,width=.3,color='#94a3b8',label='gradient 계수 p×λ')
            axes[3].set(xticks=[.18,1.18],xticklabels=['목표 1','목표 2'],ylim=(0,1.12),
                        title='④ 현재 위치에서의 gradient 가중치',ylabel='가중치')
            axes[3].legend(fontsize=8)
        if compare:
            text = ['세 방법이 고르는 방향',
                    'MGDA: QP로 매 위치의 혼합 비율 선택',
                    '  λ를 사용하지 않음. 공통 하강 방향.',
                    'LS: 사용자가 정한 λ로 가중합 최소화',
                    'STCH: λ와 현재 손실의 softmax 사용',
                    '  LS/STCH는 각 f의 동시 감소를 보장 안 함.',
                    '',f'공통 시작 θ = {vec(state["start"])}',
                    f'λ = {vec(w)}, μ = {mu.val:.3f}',
                    '모두 같은 시작 η + backtracking 사용.',
                    '반복 횟수 그래프는 실행 시간 비교가 아님.', '']
            for method in methods:
                point = state['paths'][method][-1]
                text += [f'{method}: θ = {vec(point)}', f'  (f1, f2) = {vec(objectives(point))}']
            text += ['', '참고 최적해 (볼록 예제)',f'LS θ* = {vec(w[0]*C1+w[1]*C2)}',
                     f'STCH θ* ≈ {vec(optimum)}', 'MGDA의 정지점은 출발점에 따라 달라짐.']
            notes.set_text('\n'.join(text))
        else:
            notes.set_text('1. 기준점에서의 가중 거리\n'
                f'λ = {vec(w)}, z* = {vec(IDEAL)}\n'
                f'f(θ) = {vec(f)}\n'
                'ai = λi(fi−zi*)\n'
                f'a = {vec(a)}\n\n'
                '2. max → log-sum-exp\n'
                f'TCH = max(ai) = {a.max():.4f}\n'
                f'STCH = μ log Σ exp(ai/μ) = {g:.4f}\n'
                f'0 ≤ STCH−TCH ≤ μ log 2 = {mu.val*np.log(2):.4f}\n'
                'μ↓: max에 가까워짐 / 곡률은 커질 수 있음\n\n'
                '3. 미분 가능한 gradient\n'
                'pi = exp(ai/μ) / Σ exp(aj/μ)\n'
                f'p = {vec(p)}, Σpi=1\n'
                '∇g = Σ pi λi ∇fi\n'
                f'∇g = {vec(grad)}\n'
                'piλi의 합은 일반적으로 1이 아님.\n\n'
                f'현재 θ = {vec(current)}\n'
                f'최적 θ* ≈ {vec(optimum)}\n'
                '한 걸음: θ ← θ−η∇g\n'
                '실제로 g가 줄도록 η를 자동 축소.')
        fig.canvas.draw_idle()

    def restart(_value=None):
        state['paths'] = {m: [state['start'].copy()] for m in methods}
        status.set_text('같은 출발점으로 돌아갔습니다. λ와 μ는 최적화 중에 고정합니다.')
        render()

    def advance(count):
        rates = {}
        for _ in range(count):
            for method in methods:
                point, rate = update(state['paths'][method][-1],method,weights(),mu.val,eta.val)
                state['paths'][method].append(point)
                rates[method] = rate
        status.set_text('업데이트 완료 · 마지막 실제 η: '+', '.join(f'{m}={r:.4g}' for m,r in rates.items())+' (0이면 수치적 정지)')
        render()

    def exact(_event):
        point = smooth_solution(weights(),mu.val)
        state['paths']['STCH'].append(point)
        status.set_text('STCH의 전역 최적해를 선분 위 도함수의 근으로 계산했습니다. GD 한 걸음과는 다른 직접 풀이입니다.')
        render()

    def click(event):
        if event.inaxes is axes[0] and event.button == 1 and event.xdata is not None:
            if getattr(getattr(fig.canvas,'toolbar',None),'mode',''):
                return
            state['start'] = np.array([event.xdata,event.ydata])
            restart()

    actions = [('한 걸음',lambda e:advance(1)),('100걸음',lambda e:advance(100)),
               ('출발점으로',restart)]
    if not compare:
        actions.append(('STCH 최적해로',exact))
    for i,(label,callback) in enumerate(actions):
        button=Button(fig.add_axes([.61+(i%2)*.19,.14-(i//2)*.065,.16,.043]),label)
        button.on_clicked(callback)
        buttons.append(button)
    lam.on_changed(restart)
    mu.on_changed(restart)
    fig.canvas.mpl_connect('button_press_event',click)
    render()
    plt.show()


if __name__ == '__main__':
    main()
