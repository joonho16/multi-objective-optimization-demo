"""python nonconvex_example.py: 선형 가중합이 놓치는 비볼록 Pareto front."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scalarization_core import (IDEAL, smooth_terms, nonconvex_objectives,
                                nonconvex_solutions, nonconvex_tch_solution)
from smooth_tchebycheff import style


def main():
    style()
    colors = {'LS':'#2563eb','TCH':'#9333ea','STCH':'#ea580c'}
    fig = plt.figure(figsize=(16,10),facecolor='#f5f7fb')
    grid = fig.add_gridspec(2,2,left=.055,right=.66,top=.835,bottom=.285,wspace=.34,hspace=.52)
    axes = [fig.add_subplot(grid[i,j]) for i in range(2) for j in range(2)]
    fig.suptitle('선형 가중합이 놓치는 해 · Tchebycheff는 찾을 수 있다',fontsize=19,y=.975)
    fig.text(.5,.925,r'$t\in[0,1],\quad f_1(t)=t,\quad f_2(t)=1-t^2$',ha='center',fontsize=19)
    fig.text(.5,.882,'t가 커지면 f1은 증가, f2는 감소 → 모든 t가 Pareto 해.   기준점 z*=(-0.01, -0.01)',ha='center')
    notes = fig.text(.71,.845,'',va='top',fontsize=10,linespacing=1.65)
    status = fig.text(.055,.027,'STCH 해는 [0,1]의 10,001점을 비교한 근사 전역해입니다. 비볼록 문제에서 GD의 전역 수렴을 뜻하지 않습니다.',fontsize=9,color='#475569')
    lam = Slider(fig.add_axes([.14,.165,.38,.025]),'선호도 λ1',0,1,valinit=.5,valfmt='%.2f')
    mu = Slider(fig.add_axes([.14,.10,.38,.025]),'부드러움 μ',.005,1,valinit=.1,valfmt='%.3f')
    t = np.linspace(0,1,10001)
    f = nonconvex_objectives(t)
    state = {'sweep': None}
    buttons = []

    def render():
        w = np.array([lam.val,1-lam.val])
        ls_points,tch_point,stch_point = nonconvex_solutions(w,mu.val)
        linear = f@w
        classic = np.max(w*(f-IDEAL),axis=1)
        smooth,_ = smooth_terms(f,w,mu.val)
        points = {'LS':ls_points,'TCH':np.array([tch_point]),'STCH':np.array([stch_point])}
        curves = {'LS':linear,'TCH':classic,'STCH':smooth}
        for ax in axes:
            ax.clear()
            ax.set_facecolor('white')
            ax.grid(alpha=.15)
        axes[0].plot(f[:,0],f[:,1],color='#64748b',lw=2,label='모두 Pareto 해')
        axes[0].plot([0,1],[1,0],':',color='#94a3b8',label='끝점을 잇는 직선 (중간은 불가능)')
        for name,ts in points.items():
            values = nonconvex_objectives(ts)
            axes[0].scatter(values[:,0],values[:,1],s=90 if name=='TCH' else 55,
                            marker='*' if name=='TCH' else 'o',color=colors[name],
                            label=f'{name} 해',zorder=5)
            axes[1].plot(t,curves[name],color=colors[name],label=name,lw=1.5)
            score = (nonconvex_objectives(ts)@w if name=='LS' else
                     np.max(w*(nonconvex_objectives(ts)-IDEAL),axis=1) if name=='TCH' else
                     smooth_terms(nonconvex_objectives(ts),w,mu.val)[0])
            axes[1].scatter(ts,score,color=colors[name],s=40,zorder=5)
        # LS 최소 등고선: 곡선의 내부가 아니라 끝점에 닿음.
        ls_min = min(w[0],w[1])
        if w[1] > 1e-10:
            axes[0].plot([0,1],[(ls_min-w[0]*u)/w[1] for u in (0,1)],'--',
                         color=colors['LS'],alpha=.5)
        else:
            axes[0].axvline(0,color=colors['LS'],ls='--',alpha=.5)
        # TCH sublevel의 직사각형 꼭짓점이 front의 중간에 닿는다.
        tch_min = np.max(w*(nonconvex_objectives(tch_point)-IDEAL))
        if np.all(w>0):
            corner = IDEAL+tch_min/w
            axes[0].plot([0,corner[0],corner[0]],[corner[1],corner[1],0],
                         '--',color=colors['TCH'],alpha=.55)
        axes[0].set(title='① 목적함수 공간 · 선형의 직선 vs TCH 사각형',
                    xlabel='f1=t',ylabel='f2=1−t²',xlim=(-.03,1.06),ylim=(-.03,1.06))
        axes[0].legend(fontsize=7,loc='upper right')
        axes[1].set(title='② 각 scalarization의 최소 위치 비교',xlabel='입력 t',ylabel='각 scalarization 값')
        axes[1].legend(fontsize=8)
        axes[2].plot(t,classic,color=colors['TCH'],ls='--',label='TCH: max')
        axes[2].plot(t,smooth,color=colors['STCH'],label=f'STCH: μ={mu.val:.3f}')
        axes[2].fill_between(t,classic,classic+mu.val*np.log(2),color=colors['STCH'],alpha=.08,
                             label='TCH ~ TCH+μ log 2')
        axes[2].set(title='③ max의 꺾임을 매끄럽게 만들기',xlabel='입력 t',ylabel='값 (μ가 크면 해도 달라질 수 있음)')
        axes[2].legend(fontsize=7)
        if state['sweep'] is None:
            axes[3].text(.5,.52,'「51가지 선호도로 비교」를 누르세요.\n\n선형은 두 끝점만 선택합니다.\nTCH/STCH가 선택하는 t를 함께 표시합니다.',
                         ha='center',va='center',transform=axes[3].transAxes,fontsize=11)
        else:
            preferences,all_tch,all_stch = state['sweep']
            axes[3].plot(preferences,all_tch,color=colors['TCH'],label='TCH',lw=1.5)
            axes[3].plot(preferences,all_stch,color=colors['STCH'],label='STCH',lw=1.5)
            axes[3].scatter(preferences[preferences<.5],np.ones(sum(preferences<.5)),s=12,color=colors['LS'])
            axes[3].scatter(preferences[preferences>.5],np.zeros(sum(preferences>.5)),s=12,color=colors['LS'])
            axes[3].scatter([.5,.5],[0,1],s=35,color=colors['LS'],label='LS (λ1=.5: 두 끝점)')
            axes[3].legend(fontsize=8)
        axes[3].set(title='④ 선호도마다 선택되는 해',xlabel='λ1 (λ2=1−λ1)',ylabel='최적 입력 t',
                    xlim=(-.02,1.02),ylim=(-.06,1.06))
        notes.set_text('왜 선형은 중간 해를 못 찾을까?\n'
            'gLS(t) = λ1 t + λ2(1−t²)\n'
            'gLS″(t) = −2λ2 ≤ 0\n'
            '오목한 함수의 최소는 구간 끝점에 있음.\n'
            'λ1<0.5 → t*=1, λ1>0.5 → t*=0\n'
            'λ1=0.5 → 끝점 둘만 전역 최소\n\n'
            'Tchebycheff는 가장 큰 가중 거리를 줄임\n'
            'max{λ1(t+0.01), λ2(1−t²+0.01)}\n'
            '증가하는 항과 감소하는 항의 균형점.\n'
            'λ1=λ2=0.5이면 t=1−t²\n'
            '→ t*=(√5−1)/2 ≈ 0.618034\n\n'
            f'현재 λ=({w[0]:.2f}, {w[1]:.2f}), μ={mu.val:.3f}\n'
            f'LS: t* = {", ".join(f"{x:.4f}" for x in ls_points)}\n'
            f'TCH: t* = {tch_point:.6f}\n'
            f'STCH: t* ≈ {stch_point:.4f}\n'
            f'STCH의 (f1,f2) = ({stch_point:.4f}, {1-stch_point**2:.4f})\n\n'
            'μ↓: max 값에 가까워짐.\n'
            '유한 μ에서 TCH와 해가 같을 필요는 없음.\n'
            'STCH를 전역 탐색해 선택한 해를 표시.\n'
            '이 그래프는 GD 수렴 속도 비교가 아님.')
        fig.canvas.draw_idle()

    def sweep(_event=None):
        preferences=np.linspace(0,1,51)
        rows=[nonconvex_solutions(np.array([a,1-a]),mu.val) for a in preferences]
        state['sweep']=(preferences,[r[1] for r in rows],[r[2] for r in rows])
        status.set_text(f'51가지 선호도로 전역해 비교 완료 (STCH는 간격 0.0001의 격자 근사, μ={mu.val:.3f}).')
        render()

    def change_mu(_value):
        state['sweep']=None
        status.set_text('μ가 바뀌었습니다. 선호도별 비교는 버튼을 눌러 다시 계산하세요.')
        render()

    def reset(_event):
        lam.eventson=mu.eventson=False
        lam.reset(); mu.reset()
        lam.eventson=mu.eventson=True
        state['sweep']=None
        status.set_text('λ=(0.5,0.5), μ=0.1로 초기화. LS의 끝점과 TCH/STCH의 중간 해를 비교하세요.')
        render()

    for label,callback,y in [('51가지 선호도로 비교',sweep,.15),('처음으로',reset,.085)]:
        button=Button(fig.add_axes([.62,y,.32,.045]),label)
        button.on_clicked(callback)
        buttons.append(button)
    lam.on_changed(lambda value:render())
    mu.on_changed(change_mu)
    render()
    plt.show()


if __name__ == '__main__':
    main()
