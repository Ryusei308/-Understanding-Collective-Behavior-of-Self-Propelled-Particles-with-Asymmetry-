import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML

# 濃度場を計算する関数（周期境界条件適用）
def concentration_field(mesh, n, p, v, a, b, R, k1):
    field1 = np.zeros((mesh, mesh), dtype=np.float16)  # MO field
    for i in range(n):
        px, py = p[i]
        vx, vy = v[i]
        particle_angle = np.arctan2(vy, vx)
        v2 = vx**2 + vy**2
        hill = b * v2 / (a + v2)

        x_min = max(int(px - R), 0)
        x_max = min(int(px + R), mesh)
        y_min = max(int(py - R), 0)
        y_max = min(int(py + R), mesh)

        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')

        # 距離計算に周期境界条件を適用
        dx = X - px
        dy = Y - py
        dx = np.where(dx < -mesh / 2, dx + mesh, dx)
        dx = np.where(dx > mesh / 2, dx - mesh, dx)
        dy = np.where(dy < -mesh / 2, dy + mesh, dy)
        dy = np.where(dy > mesh / 2, dy - mesh, dy)

        distance = np.sqrt(dx**2 + dy**2)
        inside_R = distance <= R
        angle_to_point = np.arctan2(dy, dx)
        angle_diff = angle_to_point - particle_angle
        f = np.exp((-1 - hill * np.cos(angle_diff)) * distance / (L / 10))

        field1[X[inside_R], Y[inside_R]] += f[inside_R] * k1

    field1[field1 < 0] = 0
    return field1

# 速度を更新する関数
def update_velocity(p, v, field1, dt, r, alpha, mesh, k2, k3):
    new_v = np.copy(v)
    for i in range(len(p)):
        px, py = round(p[i][0]), round(p[i][1])

        x_min = max(int(px - r - 2), 0)
        x_max = min(int(px + r + 2), mesh)
        y_min = max(int(py - r - 2), 0)
        y_max = min(int(py + r + 2), mesh)

        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')

        dx = X - px
        dy = Y - py
        dx = np.where(dx < -mesh / 2, dx + mesh, dx)
        dx = np.where(dx > mesh / 2, dx - mesh, dx)
        dy = np.where(dy < -mesh / 2, dy + mesh, dy)
        dy = np.where(dy > mesh / 2, dy - mesh, dy)

        distance = np.sqrt(dx**2 + dy**2)
        within_r = (distance >= r - 1) & (distance <= r + 1)
        angles = np.arctan2(dy, dx)

        forces_x = -(field1[X[within_r], Y[within_r]]) * np.cos(angles[within_r]) * k2
        forces_y = -(field1[X[within_r], Y[within_r]]) * np.sin(angles[within_r]) * k2

        new_v[i][0] += (np.sum(forces_x) * dt) - (alpha * v[i][0])
        new_v[i][1] += (np.sum(forces_y) * dt) - (alpha * v[i][1])

    return new_v

# 位置を更新する関数
def update_position(p, v, dt, mesh):
    p = p + (v * dt)
    p = np.mod(p, mesh)  # 位置をメッシュサイズで剰余計算
    return p

# オーダパラメータを計算する関数
def calculate_order_parameter(v):
    angles = np.arctan2(v[:, 1], v[:, 0])  # 速度ベクトルから角度を計算
    return np.abs(np.mean(np.exp(1j * angles)))  # オーダパラメータを計算

# 密度場を計算する関数（粒子位置を使って計算）
def calculate_density_fluctuation(p, mesh):
    density_field = np.zeros((mesh, mesh), dtype=np.float32)

    # 粒子の位置に基づいて密度場を計算
    for px, py in p:
        x, y = int(px), int(py)
        density_field[x, y] += 1

    # 密度場の平均を計算
    average_density = np.mean(density_field)

    # 密度揺らぎの計算（標準偏差）
    density_fluctuation = np.std(density_field)
    
    return density_fluctuation

# 相互作用の指標を計算する関数（濃度場の総和を計算）
def calculate_interaction_intensity(field1):
    return np.sum(field1)

# パラメータの設定
mesh = 100
dt = 0.02
a = 1
b = 0.9
k1 = 0.3
k2 = 100
L = mesh / 10
R = int(4 * round(1 / (1 - b)) * L / 10)
r = int(4 * L / 10)
alpha = 0.9
steps = int(100)
k3=0
# 粒子の初期位置と速度を指定する
n =500  # 粒子数
p = np.random.rand(n, 2) * mesh  # 初期位置をランダムに設定
angles = np.random.uniform(0, 2 * np.pi, n)  # 角度をランダムに設定
v = np.array([[np.cos(angle), np.sin(angle)] for angle in angles], dtype=np.float16)

# 濃度場とオーダパラメータの時間変化を記録
order_parameter_over_time = []
density_fluctuation_over_time = []
interaction_intensity_over_time = []

# アニメーションの初期設定
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
cax1 = ax1.imshow(np.zeros((mesh, mesh)), cmap='viridis', origin='lower', extent=(0, mesh, 0, mesh))
scat1 = ax1.scatter(p[:, 0], p[:, 1], c='k', s=mesh / 10)

# オーダパラメータのグラフ設定
line2, = ax2.plot([], [], lw=2, color='g')
ax2.set_xlim(0, steps)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Order Parameter')
ax2.set_title('Order Parameter Over Time')

# 密度揺らぎのグラフ設定
line3, = ax3.plot([], [], lw=2, color='b')
ax3.set_xlim(0, steps)
ax3.set_ylim(0, 1.5)  # 揺らぎの範囲を仮設定
ax3.set_xlabel('Time (steps)')
ax3.set_ylabel('Density Fluctuation')
ax3.set_title('Density Fluctuation Over Time')

# 新しいグラフ（相互作用指標の時間変化）
line4, = ax4.plot([], [], lw=2, color='r')
ax4.set_xlim(0, steps)
ax4.set_ylim(0, 1.2)
ax4.set_xlabel('Time (steps)')
ax4.set_ylabel('Interaction Intensity')
ax4.set_title('Interaction Intensity Over Time')

# アニメーションの更新関数
def update(step):
    global p, v, field1, order_parameter_over_time, density_fluctuation_over_time, interaction_intensity_over_time

    # 濃度場の計算
    field1 = concentration_field(mesh, n, p, v, a, b, R, k1)

    # オーダパラメータを計算
    order_parameter = calculate_order_parameter(v)
    order_parameter_over_time.append(order_parameter)

    # 密度揺らぎを計算
    density_fluctuation = calculate_density_fluctuation(p, mesh)
    density_fluctuation_over_time.append(density_fluctuation)

    # 相互作用指標を計算
    interaction_intensity = calculate_interaction_intensity(field1)
    interaction_intensity_over_time.append(interaction_intensity)

    # グラフの更新
    line2.set_data(range(len(order_parameter_over_time)), order_parameter_over_time)
    line3.set_data(range(len(density_fluctuation_over_time)), density_fluctuation_over_time)
    line4.set_data(range(len(interaction_intensity_over_time)), interaction_intensity_over_time)

    # 濃度場と粒子の位置の更新
    cax1.set_array(field1)
    cax1.set_clim(vmin=0, vmax=1)
    scat1.set_offsets(np.c_[p[:, 1], p[:, 0]])

    # 速度と位置の更新
    v = update_velocity(p, v, field1, dt, r, alpha, mesh, k2, 0)
    p = update_position(p, v, dt, mesh)

    return cax1, scat1, line2, line3, line4

# アニメーション設定
ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
html = ani.to_jshtml()
display(HTML(html))

# オーダパラメータと密度揺らぎ、相互作用指標のグラフ
fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(18, 6))

ax2.plot(range(len(order_parameter_over_time)), order_parameter_over_time, label='Order Parameter', color='g')
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Order Parameter')
ax2.set_title('Order Parameter Over Time')

ax3.plot(range(len(density_fluctuation_over_time)), density_fluctuation_over_time, label='Density Fluctuation', color='b')
ax3.set_xlabel('Time (steps)')
ax3.set_ylabel('Density Fluctuation')
ax3.set_title('Density Fluctuation Over Time')

ax4.plot(range(len(interaction_intensity_over_time)), interaction_intensity_over_time, label='Interaction Intensity', color='r')
ax4.set_xlabel('Time (steps)')
ax4.set_ylabel('Interaction Intensity')
ax4.set_title('Interaction Intensity Over Time')

plt.tight_layout()
plt.show()