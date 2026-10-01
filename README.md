# Multi-Objective Optimization 실험

## 만든 목적

다목적 최적화 논문을 읽을 때 수식만으로 이해하기 어려운 개념을 직접 조작하며 확인하기 위해 만들었습니다.
특히 “스칼라화하면 바로 해가 나오는가?”, “QP를 안 풀면 GD도 안 쓰는가?”,
“목적함수가 두 개면 해도 두 개인가?” 같은 질문을 작은 예제로 살펴봅니다.

MGDA, Linear Scalarization(LS), Tchebycheff(TCH), Smooth Tchebycheff(STCH)를
Python/Matplotlib로 시각화해, 입력의 이동과 목적함숫값의 변화를 함께 보여줍니다.
선호도에 따라 절충안이 달라지는 과정과 선형 가중합이 놓치는 Pareto 해를 눈으로 확인하는 것이 목표입니다.

## 대상과 다루는 문제

**다목적 최적화를 처음 배우거나, MGDA·Tchebycheff 관련 논문을 읽는 사람**을 대상으로 합니다.
함수와 미분의 기초를 알고 있으면, 슬라이더와 버튼으로 가중치·gradient·Pareto front의 관계를 탐색할 수 있습니다.

다루는 대상은 여러 목적함수를 같은 입력으로 평가하는 **일반적인 다목적 최적화**입니다.
모든 예제는 두 목적함수를 최소화하며, 다음 두 문제에 집중합니다.

- **입력이 2차원인 이차함수 문제**: MGDA·LS·STCH가 어느 방향으로 이동하고 어디에 도달하는지 비교합니다.
- **입력이 1차원인 비볼록 Pareto front 문제**: `t∈[0,1]`, `f1=t`, `f2=1−t²`에서 LS가 놓치는 중간 해를 확인합니다.

작은 교육용 실험으로 범위를 한정하며, 모델 학습이나 논문의 대규모 성능 벤치마크를 실행하는 도구는 아닙니다.

## 설치와 실행

Python 3와 그래픽 창을 표시할 수 있는 데스크톱 환경이 필요합니다.

```bash
git clone https://github.com/joonho16/multi-objective-optimization-demo.git
cd multi-objective-optimization-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smooth_tchebycheff.py
```

Windows PowerShell에서는 가상환경 활성화 명령을 `.venv\Scripts\Activate.ps1`로 바꿉니다.
한글이 깨지면 Noto Sans CJK KR, NanumGothic, Malgun Gothic, AppleGothic 중 하나를 설치하세요.
창이 뜨지 않는 환경에서는 GUI 지원 여부를 확인하고, 필요하면 `MPLBACKEND=QtAgg`로 실행하세요.

## 사용법

처음에는 `python nonconvex_example.py`로 1차원 예제부터 살펴보는 것을 권합니다.

1. **선호도 λ1** 슬라이더를 움직입니다. λ2는 `1−λ1`로 정해집니다.
2. LS, TCH, STCH가 고른 입력과 목적함숫값을 비교합니다.
3. **부드러움 μ**를 바꾸면서 STCH가 TCH와 어떻게 달라지는지 확인합니다.
4. **51가지 선호도로 비교**를 눌러 여러 최적해가 Pareto front에 어떻게 놓이는지 봅니다.

업데이트 과정을 보려면 `python compare_methods.py`를 실행합니다.
왼쪽 위 입력 공간을 클릭해 출발점을 바꾸고, **한 걸음** 또는 **100걸음** 버튼으로 경로를 비교하세요.
더 자세한 조작과 수식은 아래 각 예제 설명을 참고하세요.

### 실행할 화면

| 명령 | 내용 |
|---|---|
| `python MGDA.py` | MGDA와 gradient 혼합 QP |
| `python linear_scalarization.py` | 선형 가중합 |
| `python smooth_tchebycheff.py` | 같은 이차함수 예제에서 STCH, μ, softmax gradient |
| `python compare_methods.py` | 같은 출발점의 MGDA·LS·STCH 경로 및 f1/f2 변화 |
| `python nonconvex_example.py` | LS가 놓치는 중간 해를 TCH/STCH가 찾는 예제 |

## 먼저 Smooth Tchebycheff

MGDA·LS 화면과 같은 `f1=½[(x+2)²+(y+1)²]`, `f2=½[(x−1)²+(y+2)²]`를 사용합니다.
각 함수의 최솟값은 0이고, 정확한 ideal point는 `(0,0)`입니다.
코드가 사용하는 기준점은 각각에서 0.01을 뺀 utopian point `z*=(-0.01,-0.01)`입니다.
`scalarization_core.py`의 상수 이름은 `IDEAL`이지만 실제 값은 이 이동된 기준점입니다.

1. 각 목적함수에 대해 `ai=λi(fi−zi*)`를 계산합니다.
2. TCH는 `max(ai)`입니다. 가장 큰 항이 바뀌는 곳에서 미분이 안 될 수 있습니다.
3. STCH는 `μ log Σ exp(ai/μ)`로 max를 부드럽게 근사합니다.
4. `pi=softmax(a/μ)`이면 `∇g=Σ pi λi ∇fi`입니다. **λ를 한 번 더 곱해야 합니다.**
5. λ와 μ는 최적화 중에 고정하고 θ를 업데이트합니다.

`0 ≤ STCH−TCH ≤ μ log 2`입니다. μ를 줄이면 함수값이 max에 가까워집니다.
그렇다고 GD가 항상 더 빨라지는 것은 아닙니다. 작은 μ에서 곡률이 커질 수 있어,
이 구현은 실제 목적함수가 줄어들 때까지 이동폭을 반씩 줄입니다.
지수함수 계산은 가장 큰 ai를 먼저 빼서 overflow를 방지합니다.

**한 걸음 → 100걸음 → STCH 최적해로** 순서로 눌러 보세요.
최적해 버튼은 이 볼록 예제에서 두 중심을 잇는 선분 위의 도함수 근을 이분법으로 구합니다.
아래쪽 단면 그래프의 t는 이 선분을 나타내는 보조 변수이며, 원래 입력은 θ∈R²입니다.

## 세 방법 비교

- **MGDA**: 사용자의 λ 없이 매 위치에서 최소 노름 gradient 조합을 구합니다. 충분히 작은 이동으로 두 함수를 함께 줄이거나 정지합니다.
- **LS**: 사용자가 정한 λ로 `Σ λi fi`를 최소화합니다.
- **STCH**: λ를 고정해도 현재 손실에 따라 softmax 계수가 달라집니다.

비교 화면은 λ=(0.8,0.2), 같은 출발점 (0,0)으로 시작합니다. **100걸음**을 누르면
서로 다른 경로와 도달점을 볼 수 있습니다. LS/STCH는 한 목적함수를 늘리더라도
자신의 scalarization 값을 줄일 수 있습니다. MGDA의 정지점은 출발점에 따라 달라집니다.
이 화면의 가로축은 업데이트 횟수이며, 실행 시간이나 일반적인 수렴 속도 우위를 증명하지 않습니다.

## 선형 가중합이 놓치는 해

별도 문제 `t∈[0,1]`, `f1=t`, `f2=1−t²`를 사용합니다.
한 함수가 좋아지면 다른 함수가 나빠지므로 **모든 t가 Pareto 최적**입니다.

선형 가중합 `λ1 t+λ2(1−t²)`는 오목하므로 전역 최솟값은 끝점에 있습니다.
λ=(0.5,0.5)에서도 t=0과 t=1만 전역 최소이고, 내부 점은 선택되지 않습니다.

같은 선호도의 TCH는 두 항이 같아지는 `t+0.01=1−t²+0.01`, 즉 `t=1−t²`에서 최소가 되어
**t=(√5−1)/2≈0.618034**라는 중간 해를 찾습니다.
STCH도 μ=0.1에서 중간 해를 선택하며, 유한 μ에서는 TCH와 정확히 같은 해가 아닐 수 있습니다.

**51가지 선호도로 비교**를 누르면 선형은 끝점만, TCH/STCH는 중간 해들도 선택하는 것을 볼 수 있습니다.
비볼록 STCH는 10,001개 격자점 전체를 평가한 근사 전역해를 표시합니다.
이는 임의의 비볼록 문제에서 gradient descent가 전역해를 찾는다는 주장이 아닙니다.
또한 임의의 고정된 μ에서 모든 Pareto 해를 복원한다고 주장하지 않습니다.

## 개념 정리

### 해 하나와 목적함숫값 두 개

`f1(t)=t`, `f2(t)=1−t²`에서 두 함수는 **같은 변수 t**를 공유합니다.
`t=0.6`을 선택하면 해는 하나이고, 그 해의 평가 결과가 `(f1,f2)=(0.6,0.64)`입니다.
이차함수 예제에서는 입력이 `θ=(x,y)`라는 벡터이지만, 이 벡터 전체가 해 하나입니다.

- **Pareto 해**: 어떤 목적도 악화시키지 않으면서 적어도 하나를 개선할 수 없는 해.
- **Pareto set**: 입력 공간에서 Pareto 해들을 모은 집합.
- **Pareto front**: 그 해들의 목적함숫값 벡터를 모은 집합.

고정된 선호도에서 최적화해 얻은 해 하나와, 문제 전체의 Pareto 해 집합은 다릅니다.
일반적인 문제에서는 고정된 선호도에도 최적해가 여러 개일 수 있습니다.

### 선호도 λ와 함수 표기

선호도는 다음 조건을 만족하는 가중치 벡터입니다. 이 집합을 simplex라고 합니다.

$$
\lambda_i\geq0,\qquad\sum_{i=1}^{m}\lambda_i=1.
$$

`g(x | λ)`는 이 문맥에서 **λ를 정해 놓고 x를 바꿔 평가하는 함수**입니다.
세로줄은 파라미터의 역할을 강조하는 관례이며, 일반 함수에서 보편적으로 강제되는 규칙은 아닙니다.
저자의 정의와 최적화 변수 표시를 함께 확인해야 합니다.

| 표기 | 이 문맥의 의미 | 영어로 읽는 예 |
|---|---|---|
| `g(x \| λ)` | λ가 주어진 상태의 함수 | g of x given lambda |
| `g(x; λ)` | x와 파라미터 λ의 역할을 구분 | g of x with parameter lambda |
| `g(x, λ)` | 두 입력을 나란히 표시 | g of x and lambda |

### 스칼라화, GD, QP는 서로 다른 개념

| 개념 | 역할 |
|---|---|
| 스칼라화 | 여러 목적함수를 목적함수 하나로 바꿈 |
| GD(경사하강법) | gradient로 입력을 반복 갱신하는 풀이 방법 |
| QP(이차계획 문제) | 이차 목적함수와 선형 제약을 가진 최적화 문제의 형태 |

선형 스칼라화는 다음 문제를 만듭니다.

$$
\min_{x\in X} g(x\mid\lambda),\qquad
g(x\mid\lambda)=\sum_i\lambda_i f_i(x).
$$

가중합을 만들었다고 즉시 최적해가 나오는 것은 아닙니다. GD, 해석적 계산 등으로 이 문제를 풀어야 합니다.
여기서 linear는 **목적함숫값을 선형으로 결합한다**는 뜻이지, 각 목적함수가 x에 대해 선형이라는 뜻은 아닙니다.
λ를 바꾼 뒤 다시 최적화하면 다른 절충안을 얻을 수 있지만, 같은 해가 반복될 수도 있습니다.
선형 가중합으로는 비볼록 Pareto front의 일부 해를 얻지 못할 수 있습니다.

MGDA는 현재 위치의 gradient들을 결합하기 위해 매 반복에서 다음 QP를 풉니다.

$$
\min_{\alpha_i\geq0,\;\sum_i\alpha_i=1}
\left\|\sum_i\alpha_i\nabla f_i(x)\right\|^2.
$$

이 α는 사용자가 지정하는 선호도 λ와 다릅니다. 구한 방향으로 입력을 업데이트합니다.
이 저장소는 목적이 두 개이므로 α의 해를 공식과 구간 제한으로 계산하며, 별도 QP 솔버가 필요하지 않습니다.
LS/STCH는 이러한 gradient 혼합 QP를 매번 풀 필요가 없지만, GD로 최적화할 수 있습니다.
이 예제의 LS 자체도 이차함수이므로, **MGDA 내부의 QP를 생략한다는 것과 목적함수가 이차식인지 여부는 별개**입니다.

### Ideal point, ε, Tchebycheff

최솟값이 존재할 때 ideal point는 각 목적을 따로 최소화한 값을 모은 기준점입니다.

$$
z_i^{\mathrm{ideal}}=\min_{x\in X}f_i(x).
$$

`t∈[0,1]`에서 f1은 t=0에서, f2는 t=1에서 각각 최솟값 0을 얻습니다.
따라서 ideal point는 `(0,0)`입니다. 두 값을 동시에 0으로 만드는 입력은 없어도 됩니다.

이 저장소에서는 ε=0.01을 사용해 기준점을 더 작은 쪽으로 옮깁니다.

$$
z_i^*=z_i^{\mathrm{ideal}}-\epsilon,\qquad\epsilon>0.
$$

이것이 utopian point입니다. ε는 기준점의 이동량이고, μ는 아래에서 설명할 매끄러움의 정도입니다.
정확한 ideal point를 쓰는 정의도 있으므로 ε를 항상 넣어야 하는 것은 아닙니다.

기본 가중 Tchebycheff는 가장 큰 가중 편차를 최소화합니다.

$$
\min_{x\in X}\max_i\{\lambda_i(f_i(x)-z_i^*)\}.
$$

이 기준점은 모든 가능한 목적함숫값보다 작으므로 절댓값을 넣어도 같은 식입니다.
동일 가중치 λ=(0.5,0.5)에서는:

$$
t+\epsilon=1-t^2+\epsilon
\quad\Longrightarrow\quad
t^*=\frac{\sqrt5-1}{2}\approx0.618034.
$$

목적함숫값 벡터도 약 `(0.618034,0.618034)`입니다.
이 ε의 상쇄는 동일 가중치와 동일 이동량인 경우이며, 다른 가중치에서는 ε가 해에 영향을 줄 수 있습니다.

### 뾰족점과 Smooth Tchebycheff

`max`에서 최대인 항이 바뀌는 지점은 미분이 불가능할 수 있습니다.
그렇다고 최적화가 불가능한 것은 아닙니다. 구조에 따라 부분기울기 등 비매끄러운 최적화 방법을 사용할 수 있습니다.
다만 매끄러운 함수에 대한 GD의 수렴 보장을 그대로 적용할 수는 없습니다.

STCH는 max를 다음 log-sum-exp로 매끄럽게 근사합니다.

$$
g_\mu(x\mid\lambda)=\mu\log\sum_i
\exp\left(\frac{\lambda_i(f_i(x)-z_i^*)}{\mu}\right),\qquad\mu>0.
$$

원래 목적함수들이 매끄러우면 이 함수도 매끄러워 gradient 기반 최적화를 적용할 수 있습니다.
논문의 “fast convergence rate for the gradient-based method”는
**gradient 기반 방법으로 풀 때 빠르게 수렴한다**는 뜻입니다. GD를 쓰지 않는다는 뜻이 아닙니다.
구체적인 수렴 보장은 논문의 가정에 따르며, 작은 μ나 특정 방법이 항상 더 짧은 실행 시간을 보장하지 않습니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `MGDA.py` | 두 gradient의 혼합과 QP 해, 입력 업데이트 시각화 |
| `linear_scalarization.py` | 가중합과 이차함수 예제의 해석적 최적해 |
| `smooth_tchebycheff.py` | STCH와 세 방법 비교 화면 |
| `compare_methods.py` | 비교 화면 실행 진입점 |
| `nonconvex_example.py` | 비볼록 Pareto front에서 LS/TCH/STCH 비교 |
| `scalarization_core.py` | 공유 목적함수, gradient, 최적해 및 업데이트 계산 |
| `requirements.txt` | NumPy, Matplotlib, PyQt6 의존성 |

## 참고 문헌

[Lin et al., ICML 2024 — Smooth Tchebycheff Scalarization for Multi-Objective Optimization](https://proceedings.mlr.press/v235/lin24y.html).
이 저장소는 개념을 이해하기 위한 작은 예제이며, 논문의 전체 실험을 재현하는 코드는 아닙니다.
