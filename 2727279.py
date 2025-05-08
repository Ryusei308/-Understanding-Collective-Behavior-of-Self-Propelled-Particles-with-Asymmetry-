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
        dx = np.where(px-dx < 0, px - dx + mesh, dx)       
        dx = np.where(px-dx>mesh,px-dx-mesh,dx)
        dx = np.where(0<=px-dx,X - px,dx)
        dx = np.where(px-dx<=mesh,X - px,dx)
        dy = np.where(py-dy < 0, py - dy + mesh, dy)
        dy = np.where(py-dy>mesh,py-dy-mesh,dy)
        dy = np.where(0<=py-dy,Y - py,dy)
        dy = np.where(py-dy<=mesh,Y - py,dy)

        distance = np.sqrt(dx**2 + dy**2)

        inside_R = distance <= R
        angle_to_point = np.arctan2(dy, dx)
        angle_diff = angle_to_point - particle_angle
        f = np.exp((-1 - hill * np.cos(angle_diff)) * distance / (L / 10))

        field1[X[inside_R], Y[inside_R]] += f[inside_R] * k1

    field1[field1 < 0] = 0
    field1 = apply_periodic_boundary(field1, mesh)
    return field1

def apply_periodic_boundary(field, mesh):
    # 左右の境界を反対側にコピー
    field[:, 0] += field[:, -1]  # 左端に右端をコピー
    field[:, -1] += field[:, 0]  # 右端に左端をコピー

    # 上下の境界を反対側にコピー
    field[0, :] += field[-1, :]  # 上端に下端をコピー
    field[-1, :] += field[0, :]  # 下端に上端をコピー

    return field

# 速度を更新する関数（周期境界条件適用）
def update_velocity(p, v, field1, dt, r, alpha, mesh, k2, k3):
    new_v = np.copy(v)
    for i in range(len(p)):
        px, py = round(p[i][0]), round(p[i][1])

        x_min = max(int(px - r - 2), 0)
        x_max = min(int(px + r + 2), mesh)
        y_min = max(int(py - r - 2), 0)
        y_max = min(int(py + r + 2), mesh)

        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')

        # 距離計算に周期境界条件を適用
        dx = X - px
        dy = Y - py
        dx = np.where(px-dx < 0, px - dx + mesh, dx)       
        dx = np.where(px-dx>mesh,px-dx-mesh,dx)
        dx = np.where(0<=px-dx,X - px,dx)
        dx = np.where(px-dx<=mesh,X - px,dx)
        dy = np.where(py-dy < 0, py - dy + mesh, dy)
        dy = np.where(py-dy>mesh,py-dy-mesh,dy)
        dy = np.where(0<=py-dy,Y - py,dy)
        dy = np.where(py-dy<=mesh,Y - py,dy)

        distance = np.sqrt(dx**2 + dy**2)

        within_r = (distance >= r-1) & (distance <= r+1)
        angles = np.arctan2(dy, dx)

        forces_x = -(field1[X[within_r], Y[within_r]]) * np.cos(angles[within_r]) * k2
        forces_y = -(field1[X[within_r], Y[within_r]]) * np.sin(angles[within_r]) * k2

        new_v[i][0] += (np.sum(forces_x) * dt) - (alpha * v[i][0])
        new_v[i][1] += (np.sum(forces_y) * dt) - (alpha * v[i][1])

    return new_v

# 位置を更新する関数（周期境界条件適用）
def update_position(p, v, dt, mesh):
    p = p + (v * dt)

    # 位置をメッシュサイズで剰余計算（移動しすぎた場合）
    p = np.mod(p, mesh)
   
    return p

# オーダパラメータを計算する関数
def calculate_order_parameter(v):
    angles = np.arctan2(v[:, 1], v[:, 0])  # 速度ベクトルから角度を計算
    return np.abs(np.mean(np.exp(1j * angles)))  # オーダパラメータを計算

# パラメータの設定
mesh = 100
dt = 0.02
a = 1
b = 0.8
k1 = 0.3
k2 = 100
L = mesh / 10
R = int(4 * round(1/(1 - b)) * L / 10)
r = int(4 * L / 10)
k3 = 0
alpha = 0.08
steps = int(200)

# 粒子の初期位置と速度を指定する
n = 200  # 粒子の数を2に変更
#p = np.array([[40, 50], [ 30, 60]])  # 初期位置を指定
#angles = [math.radians(190), math.radians(220)]  # 角度を指定（45度と225度）
#p = np.array([[10, 50], [ 30, 60]])  # 初期位置を指定
#angles = [math.radians(20), math.radians(220)]  # 角度を指定（45度と225度）
#p = np.array([[22, 50], [ 30, 60]])  # 初期位置を指定
#angles = [math.radians(5), math.radians(270)]  # 角度を指定（45度と225度）
p = np.random.rand(n, 2) * mesh  # 初期位置をランダムに設定
angles = np.random.uniform(0, 2 * np.pi, n)  # 角度をランダムに設定


# 指定した角度で速度を設定
v = np.array([[np.cos(angle), np.sin(angle)] for angle in angles], dtype=np.float16)

# 濃度の時間変化とオーダパラメータの時間変化を保存するリスト
concentration_over_time = []
order_parameter_over_time = []

# アニメーションの初期設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
cax1 = ax1.imshow(np.zeros((mesh, mesh)), cmap='viridis', origin='lower', extent=(0, mesh, 0, mesh))
scat1 = ax1.scatter(p[:, 0], p[:, 1], c='k', s=mesh/100)

# オーダパラメータのグラフ設定
line2, = ax2.plot([], [], lw=2, color='g')
ax2.set_xlim(0, steps)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Order Parameter')
ax2.set_title('Order Parameter Over Time')

# アニメーションの更新関数
def update(step):
    global p, v, field1, concentration_over_time, order_parameter_over_time
    
    # 濃度場の計算
    field1 = concentration_field(mesh, n, p, v, a, b, R, k1)
    
    # オーダパラメータを計算
    order_parameter = calculate_order_parameter(v)
    order_parameter_over_time.append(order_parameter)
    
    # グラフの更新
    line2.set_data(range(len(order_parameter_over_time)), order_parameter_over_time)
    
    # 濃度場と粒子の位置の更新
    cax1.set_array(field1)
    cax1.set_clim(vmin=0, vmax=1)
    scat1.set_offsets(np.c_[p[:, 1], p[:, 0]])

    # 速度と位置の更新
    v = update_velocity(p, v, field1, dt, r, alpha, mesh, k2, k3)
    p = update_position(p, v, dt, mesh)

    return cax1, scat1, line2

# アニメーション設定
ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
html = ani.to_jshtml()
display(HTML(html))

# オーダパラメータの時間変化のグラフも別途表示
fig2, ax2 = plt.subplots()
ax2.plot(range(len(order_parameter_over_time)), order_parameter_over_time, label='Order Parameter Over Time', color='g')
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Order Parameter')
ax2.set_title('Order Parameter Over Time')
plt.show()
