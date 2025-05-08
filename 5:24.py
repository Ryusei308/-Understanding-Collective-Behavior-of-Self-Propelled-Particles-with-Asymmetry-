import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython import display
from matplotlib.animation import FuncAnimation

# 定数の設定
u = 0.9  # 摩擦係数
m = 0.8  # 粒子の重さ
a = 0.59  # 濃度関数の定数
r = 1.5 # 粒子の中心からの距離
b = 10# 粒子の数
p=400
# 初期位置と初速度の設定（ベクトルとして、浮動小数点数で指定）
initial_positions = np.random.rand(b, 2) * p  # 粒子の初期位置 (0から10000の間のランダムな座標)
initial_velocities = np.random.randn(b, 2) * 0.01  # 粒子の初期速度 (標準正規分布から生成)

# 時間の設定
t = np.linspace(0, 30, 100)

# 運動方程式の定義
def new_model(current_position, x_range, y_range):
    def c(θ, l, v):
        def b(v):
            return abs(v) / (1 + abs(v))

        return 1 - a * np.sum(np.exp(-(1 + b(v) * np.cos(θ)) * l))
        
    
    def model(v, t):
        l = np.sqrt((current_position[0] - x_range[:, np.newaxis])**2 + (current_position[1] - y_range[:, np.newaxis])**2)
        θ = np.arctan2(y_range[:, np.newaxis] - current_position[1], x_range[:, np.newaxis] - current_position[0]) - np.arctan2(v[1], v[0])  # 初期速度の方向を補正
        
        def force_components(v, θ, l):
          #Fx = np.zeros_like(v[:, 0])
          #Fy = np.zeros_like(v[:, 1])
          Fx = 0
          Fy = 0
          for k in range(b):
            for i in range(len(θ[k])):
                F = r * c(θ[k][i], 2, v[k])
                
                Fx += (F * np.cos(θ[k][i]))
                Fy += (F * np.sin(θ[k][i]))
                
            return Fx, Fy
        Fx, Fy = force_components(v, θ, 2)

        dvxdt = -u * v[0] + Fx / m
        dvydt = -u * v[1] + Fy / m
        #print(dvxdt)
        
        return np.ravel([dvxdt, dvydt])
       
    return model



# 運動方程式の数値解法
def solve_motion(model, initial_velocity, t):
    sol = odeint(model, initial_velocity, t)
    #print(sol)
    return sol

# 粒子の速度記録用のリスト
velocities_record = [[] for _ in range(len(initial_positions))]

# アニメーションの設定
fig, ax = plt.subplots()
lines = [ax.plot([], [], 'bo-')[0] for _ in range(len(initial_positions))]  # 各粒子の軌跡をプロットするためのライン

def init():
    ax.set_xlim(0, p)
    ax.set_ylim(0, p)
    return lines

def update(frame):
    global current_position, sol,initial_velocities,initial_positions
    if frame == 0:
      sol = []
    for i, line in enumerate(lines):
        current_position = initial_positions[i]
        if frame == 0:
          sol.append(solve_motion(new_model(current_position, np.linspace(-p, p, 12), np.linspace(-p, p, 12)), initial_velocities[i], t))
        initial_theta = np.arctan2(sol[i][frame, 1], sol[i][frame, 0])
        current_position[0] += sol[i][frame, 0] * t[frame]
        current_position[1] += sol[i][frame, 1] * t[frame]

        x = current_position[0] % p
        y = current_position[1] % p
        #ax.scatter(x, y, color='red', s=50)

        line.set_data(x, y)

        #initial_directions = np.arctan2(sol[i][frame,1], sol[i][frame,0])

        velocities_record[i].append(np.linalg.norm(sol[i][frame]))

    return lines

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
html = display.HTML(ani.to_jshtml())
display.display(html)
plt.close()

# 粒子ごとの速度の変化をプロット
for i, velocities in enumerate(velocities_record):
    mean_velocities = np.mean(velocities_record, axis=0)
plt.plot(t, mean_velocities, label='Mean Velocity', color='red')

    #plt.plot(t, velocities)
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity vs Time for Each Particle')
plt.show()

# 初期移動方向の計算
# 初期移動方向の計算
initial_directions = np.arctan2(initial_velocities[:, 1], initial_velocities[:, 0])

# ヒストグラムの作成
plt.hist(initial_directions, bins=np.linspace(-np.pi, np.pi, 21), density=True, alpha=0.7, color='blue')
plt.xlabel('Initial Direction (radians)')
plt.ylabel('Frequency')
plt.title('Histogram of Initial Directions')
plt.show()


# 初期速度のヒストグラムのプロット
# 初期速度のヒストグラムのプロット
plt.hist(initial_velocities[:,0], bins=20, density=True, alpha=0.7, color='green', label='Component 1')
plt.hist(initial_velocities[:,1], bins=20, density=True, alpha=0.7, color='blue', label='Component 2')
plt.xlabel('Initial Velocity')
plt.ylabel('Frequency')
plt.title('Histogram of Initial Velocities')
plt.legend()
plt.show()

