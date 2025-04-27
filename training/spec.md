# Mario-v0

``` python
"""
A 代表跳躍, B 代表加速, from NES controll
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

observation_space: (240, 256, 3), uint8, (高, 寬, RGB)

reward components
    clip into <-15, 15>
    1. 水平位移差： x_t+1 - x_t
    2. every step negative reward
    3. death: -15

info{
  'coins': int,     # 收集到的金幣數
  'flag_get': bool, # 是否到達旗幟
  'life': int,      # 剩餘命數 (3/2/1)
  'score': int,     # 累積分數
  'stage': int,     # 關卡編號 (1-4)
  'status': str,    # Mario 狀態：'small'/'tall'/'fireball'
  'time': int,      # 剩餘時間
  'world': int,     # 世界編號 (1-8)
  'x_pos': int,     # 水平位置
  'y_pos': int,     # 垂直位置
}
"""
```