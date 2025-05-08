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
        # 距離計算に周期境界条件を適用

        dx = np.where(px-dx < 0, px - dx + mesh, dx)       
        dx = np.where(px-dx>mesh,px-dx-mesh,dx)
        dx = np.where(0<=px-dx,X - px,dx)
        dx = np.where(px-dx<=mesh,X - px,dx)
        dy = np.where(py-dy < 0, py - dy + mesh, dy)
        dy = np.where(py-dy>mesh,py-dy-mesh,dy)
        dy = np.where(0<=py-dy,Y - py,dy)
        dy = np.where(py-dy<=mesh,Y - py,dy)

          
        #dx = (dx + mesh/2) % mesh - mesh/2
        #dy = (dy + mesh/2) % mesh - mesh/2
        distance = np.sqrt(dx**2 + dy**2)

        inside_R = distance <= R
        angle_to_point = np.arctan2(dy, dx)
        angle_diff = angle_to_point - particle_angle
        f = np.exp((-1 - hill * np.cos(angle_diff)) * distance / (L / 10))

        field1[X[inside_R], Y[inside_R]] += f[inside_R] * k1

    field1[field1 < 0] = 0
    field1 = apply_periodic_boundary(field1, mesh)
    return field1
#def apply_periodic_boundary(field, mesh):
    # 左右の境界を反対側にコピー
    #field[:, :2] += field[:, -2:]  # 左端に右端をコピー
    #field[:, -2:] += field[:, :2]  # 右端に左端をコピー

    # 上下の境界を反対側にコピー
    #field[:2, :] += field[-2:, :]  # 上端に下端をコピー
    #field[-2:, :] += field[:2, :]  # 下端に上端をコピー

    #return field
def apply_periodic_boundary(field, mesh):
    # 左右の境界を反対側にコピー
    field[:, 0] = field[:, -1]  # 左端に右端をコピー
    field[:, -1] = field[:, 0]  # 右端に左端をコピー

    # 上下の境界を反対側にコピー
    field[0, :] = field[-1, :]  # 上端に下端をコピー
    field[-1, :] = field[0, :]  # 下端に上端をコピー

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

        #dx = np.where(px-dx < 0, px - dx + mesh, dx)
        #dx = np.where(px-dx>mesh,px-dx-mesh,dx)
        #dy = np.where(py-dy < 0, py - dy + mesh, dy)
        #dy = np.where(py-dy>mesh,py-dy-mesh,dy)
        #dx = np.where(px-dx < 0, px - dx + mesh, dx)
        #dx = np.where(px-dx>mesh,px-dx-mesh,dx)
        #dy = np.where(dy < 0, py - dy + mesh, dy)
        #dy = np.where(py-dy>mesh,py-dy-mesh,dy)

        #dx = (dx + mesh/2) % mesh - mesh/2
        #dy = (dy + mesh/2) % mesh - mesh/2
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

# パラメータの設定
mesh = 100
n = 200  # 粒子の数
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
steps = int(100)

# 粒子の初期位置と速度をランダムに設定
p = np.random.rand(n, 2) * mesh  # 位置をランダムに設定
theta = np.random.rand(n) * 2 * np.pi
v = np.array([[np.cos(theta[i]), np.sin(theta[i])] for i in range(n)], dtype=np.float16)

# 濃度の時間変化を保存するリスト
concentration_over_time = []

# 粒子の周囲の濃度を平均する関数
def calculate_average_concentration_around_particle(particle_index, radius, field1):
    px, py = p[particle_index]
    
    # 粒子の周囲の領域を取得
    x_min = max(int(px - radius), 0)
    x_max = min(int(px + radius), mesh)
    y_min = max(int(py - radius), 0)
    y_max = min(int(py + radius), mesh)

    # 濃度場からその領域を抽出
    local_concentration = field1[x_min:x_max, y_min:y_max]
    avg_concentration = np.mean(local_concentration)
    
    return avg_concentration

# アニメーションの初期設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
cax1 = ax1.imshow(np.zeros((mesh, mesh)), cmap='viridis', origin='lower', extent=(0, mesh, 0, mesh))
scat1 = ax1.scatter(p[:, 0], p[:, 1], c='k', s=mesh/100)
particle_scat = ax1.scatter([], [], c='r', s=mesh/50)  # 特定の粒子を赤で表示

# 時間に対する濃度変化のグラフ設定
line, = ax2.plot([], [], lw=2)
ax2.set_xlim(0, steps)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Average Concentration')
ax2.set_title('Average Concentration Around Particle')

# アニメーションの更新関数
def update(step):
    global p, v, field1, concentration_over_time
    
    # 濃度場の計算
    field1 = concentration_field(mesh, n, p, v, a, b, R, k1)
    
    # 特定の粒子周囲の平均濃度を計算
    avg_concentration = calculate_average_concentration_around_particle(0, 20, field1)
    concentration_over_time.append(avg_concentration)
    
    # グラフの更新
    line.set_data(range(len(concentration_over_time)), concentration_over_time)
    
    # 濃度場と粒子の位置の更新
    cax1.set_array(field1)
    cax1.set_clim(vmin=0, vmax=1)
    scat1.set_offsets(np.c_[p[:, 1], p[:, 0]])
    particle_scat.set_offsets([p[0, 1], p[0, 0]])  # 0番目の粒子を強調表示
    
    # 速度と位置の更新
    v = update_velocity(p, v, field1, dt, r, alpha, mesh, k2, k3)
    p = update_position(p, v, dt, mesh)

    return cax1, scat1, particle_scat, line

# アニメーション設定
ani = animation.FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
html = ani.to_jshtml()
display(HTML(html))

# 時間に対する濃度変化のグラフも別途表示
fig2, ax2 = plt.subplots()
ax2.plot(range(len(concentration_over_time)), concentration_over_time, label='Average Concentration around Particle')
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Average Concentration')
ax2.set_title('Average Concentration Around Particle Over Time')
plt.show()