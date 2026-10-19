# -*- coding: utf-8 -*-
"""
kf_three_rounds.py
三次(滚动)集合卡尔曼滤波(EnKF)校正 + 结果评估 —— Python版
==================================================
输入: 两列CSV/TXT (obs, topmodel), 第1行为标题, 如 001.txt
方法: 随机集合Kalman滤波 (《集合卡尔曼文档》式23-30, N个成员, 扰动观测)
  状态: 流量校正偏差 d(t), 随机游走模型
    预报:  d_f = d_a ;  P_f = P_a + Q          (式23, 过程噪声Q)
    增益:  K   = P_f / (P_f + R)               (式28)
    分析:  d_a = d_f + K*(y - (sim + d_f))     (式29)
           P_a = (1 - K) * P_f                 (式30)
    校正流量 = sim + d_a ;  观测算子 H = 1
  三次滚动: 第r轮以 R_r = R0/r 重新运行整场
            (轮次越后越信任观测, 等效增益逐轮增大, 与校正1.xlsx规律一致)
输出: 控制台指标表(FR/FP/FT/NSE) + 结果CSV + 对比图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = {
    'dataFile': '006.txt',      # 输入: obs, topmodel 两列
    'eventName': '20190722',
    'dt' : 1.0,                  # 时间步长 1 h
    'Q'  : 25.0,                 # 过程噪声方差 (m^3/s)^2
    'R0' : 50.0,                 # 第1轮观测噪声方差 (m^3/s)^2
    'P0' : 100.0,                # 初始偏差方差
    'nRounds': 3,
    'N'  : 100,                 # 集合数
    'seed': 2024,               # 随机种子(保证可复现)
    'outFile': 'kf_three_rounds_result.csv',
    'figFile': 'kf_three_rounds_result.png',
}


def enkf_round(obs, sim, R, cfg, rng):
    """单轮随机集合Kalman滤波(stochastic EnKF):
       N个成员传播偏差状态, 样本协方差估计P (式26-27),
       扰动观测分析 (式29); 集合均值作为校正结果。"""
    N, T = cfg['N'], len(obs)
    d = rng.normal(0.0, np.sqrt(cfg['P0']), N)          # 初始集合
    u = np.empty(T); Khist = np.empty(T)
    for t in range(T):
        d += rng.normal(0.0, np.sqrt(cfg['Q']), N)      # 成员预报 (式23)
        P = float(np.var(d, ddof=1))                    # 样本协方差 (式26-27)
        K = P / (P + R)                                 # Kalman增益 (式28)
        z = obs[t] + rng.normal(0.0, np.sqrt(R), N)     # 扰动观测
        d = d + K * (z - (sim[t] + d))                  # 分析 (式29)
        u[t] = sim[t] + d.mean()                        # 集合均值
        Khist[t] = K
    return u, Khist


def metrics(obs, q, dt):
    """与文档表4一致的指标(英文列名)"""
    Wo, Ws = obs.sum()*0.36*dt, q.sum()*0.36*dt
    FR = (Ws-Wo)/Wo*100
    Qp, i1 = obs.max(), int(obs.argmax())
    Qps, i2 = q.max(), int(q.argmax())
    FP = (Qps-Qp)/Qp*100
    FT = (i2-i1)*dt
    NSE = 1 - np.sum((obs-q)**2)/np.sum((obs-obs.mean())**2)
    return dict(Wobs=Wo, Wsim=Ws, FR=FR, Qpk=Qp, Qps=Qps,
                FP=FP, FT=FT, NSE=NSE)


def main(cfg):
    rng = np.random.default_rng(cfg['seed'])
    df = pd.read_csv(cfg['dataFile'])
    obs = df.iloc[:, 0].to_numpy(float)
    sim = df.iloc[:, 1].to_numpy(float)
    T = len(obs); t = np.arange(1, T+1)*cfg['dt']

    HDR = ['Scheme','Simulated flood volume 10^4 m3','Flood volume relative error FR/%',
           'Observed peak discharge (m3/s)','Simulated peak discharge (m3/s)',
           'Peak discharge relative error FP/%','Peak time error FT/h',
           'Nash-Sutcliffe efficiency NSE']
    print(f"\n===== {cfg['eventName']} | 三次EnKF (N={cfg['N']}, Q={cfg['Q']}, R0={cfg['R0']}, seed={cfg['seed']}) =====")
    rows = []
    m0 = metrics(obs, sim, cfg['dt'])
    rows.append(['Baseline TOPMODEL Sim.', m0['Wsim'], m0['FR'], m0['Qpk'], m0['Qps'], m0['FP'], m0['FT'], m0['NSE']])

    res, Kh = [], []
    for r in range(1, cfg['nRounds']+1):
        Rr = cfg['R0'] / r
        u, Khist = enkf_round(obs, sim, Rr, cfg, rng)
        res.append(u); Kh.append(Khist)
        m = metrics(obs, u, cfg['dt'])
        rows.append([f'{r}th Rolling Update', m['Wsim'], m['FR'], m['Qpk'], m['Qps'], m['FP'], m['FT'], m['NSE']])

    tbl = pd.DataFrame(rows, columns=HDR)
    print(tbl.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))

    tbl.to_csv(cfg['outFile'], index=False)
    detail = pd.DataFrame({'obs': obs, 'sim': sim,
                           **{f'round{r}_upd': res[r-1] for r in range(1, cfg['nRounds']+1)},
                           **{f'round{r}_K': Kh[r-1] for r in range(1, cfg['nRounds']+1)}})
    detail.to_csv(cfg['outFile'].replace('.csv', '_detail.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(t, obs, 'k-', lw=1.6, label='实测')
    ax.plot(t, sim, '--', lw=1.2, label='TOPMODEL')
    for r, s in zip(range(1, cfg['nRounds']+1), ['-', '-.', ':']):
        ax.plot(t, res[r-1], s, lw=1.4, label=f'第{r}轮校正')
    ax.set(xlabel='时间步 (h)', ylabel='流量 (m$^3$/s)', title=f"{cfg['eventName']} 三次Kalman滤波校正")
    ax.grid(alpha=.3); ax.legend(loc='best')
    ax = axes[1]
    for r in range(1, cfg['nRounds']+1):
        ax.plot(t, Kh[r-1], lw=1.3, label=f'第{r}轮 K')
    ax.set(xlabel='时间步 (h)', ylabel='Kalman增益 K', title='逐时刻增益(0<K<1)', ylim=(0, 1.05))
    ax.grid(alpha=.3); ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(cfg['figFile'], dpi=300, bbox_inches='tight')
    print(f"\n结果CSV: {cfg['outFile']}  图: {cfg['figFile']}")


if __name__ == '__main__':
    main(CFG)