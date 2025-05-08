import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython import display
from matplotlib.animation import FuncAnimation

# 定数の設定
u = 0.99  # 摩擦係数
m = 1  # 粒子の重さ
a = 0.59  # 濃度関数の定数
r = 1  # 粒子の中心からの距離

# 初期位置と初速度の設定（ベクトルとして、浮動小数点数で指定）
initial_position = np.array([500.0, 500.0])
initial_velocity = np.array([-1.0, 7.0])  # 初期速度を設定

# 時間の設定
t = np.linspace(0, 30, 100)

# 運動方程式の定義
def new_model(current_position, x_range, y_range):
    def c(θ, l, v):
        def b(v):
            return abs(v) / (1 + abs(v))
        return 1 - a * np.exp(-(1 + b(v) * np.cos(θ)) * l)

    def model(v, t):
        l = np.sqrt((current_position[0] - x_range[:, np.newaxis])**2 + (current_position[1] - y_range[:, np.newaxis])**2)
        θ = np.arctan2(y_range[:, np.newaxis] - current_position[1], x_range[:, np.newaxis] - current_position[0]) - np.arctan2(initial_velocity[1], initial_velocity[0])  # 初期速度の方向を補正
        F = np.sum(r * c(θ, l, v))
        dvdt = (-u * v + F) / m
        return dvdt
    
    return model

# 運動方程式の数値解法
def solve_motion(model, initial_velocity, t):
    sol = odeint(model, initial_velocity, t)
    return sol

# アニメーションの設定
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-')

def init():
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    return line,

def update(frame):
    global current_position, initial_theta
    if frame == 0:
        current_position = initial_position.copy()
        initial_theta = np.arctan2(initial_velocity[1],initial_velocity[0])  # 初期速度の角度を取得
    
    current_position[0] += sol[frame, 0] * np.cos(initial_theta)
    current_position[1] += sol[frame, 1] * np.sin(initial_theta)

    x = current_position[0] % 1000 - 500
    y = current_position[1] % 1000 - 500

    ax.scatter(x, y, color='red', s=50)
    line.set_data(x, y)

    return line,

# 運動方程式を関数に設定
model = new_model(initial_position, np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
sol = solve_motion(model, initial_velocity, t)

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
html = display.HTML(ani.to_jshtml())
display.display(html)
plt.figure()
plt.plot(t, sol, label='Velocity')

plt.title('Velocity Over Time')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.tight_layout()

plt.show()

