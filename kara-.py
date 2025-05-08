import numpy as np
import matplotlib.pyplot as plt

# 提供された関数のインポート
def concentration_field(mesh, n, p, v, a, b, R, k1, L):
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
def update_velocity_with_gradient(p, v, field1, dt, alpha, mesh, k2):
    new_v = np.copy(v)
    grad_x, grad_y = np.gradient(field1)
    for i in range(len(p)):
        px, py = round(p[i][0]), round(p[i][1])
        local_grad_x = grad_x[px % mesh, py % mesh]
        local_grad_y = grad_y[px % mesh, py % mesh]
        force_x = -k2 * local_grad_x
        force_y = -k2 * local_grad_y
        new_v[i][0] += (force_x * dt) - (alpha * v[i][0])
        new_v[i][1] += (force_y * dt) - (alpha * v[i][1])
    return new_v

# 位置を更新する関数
def update_position(p, v, dt, mesh):
    new_p = p + (v * dt)
    return np.mod(new_p, mesh)

# オーダパラメータ計算
def calculate_order_parameter(v):
    angles = np.arctan2(v[:, 1], v[:, 0])  # 速度ベクトルから角度を計算
    return np.abs(np.mean(np.exp(1j * angles)))  # オーダパラメータを計算

# パラメータ範囲
angle_range = np.linspace(-90, 90, 10)  # 入射角 [-90°, 90°]
y_offset_range = np.linspace(-10, 10, 10)  # 粒子間y方向のズレ

# データ保存用
order_parameters = np.zeros((len(angle_range), len(y_offset_range)))

# シミュレーションパラメータ
mesh = 100
dt = 0.03
a, b = 1, 0.95
k1, k2 = 0.3, 3000
L = mesh / 10
R = int(4 * round(1/(1 - b)) * L / 10)
r = int(4 * L / 10)
k3, alpha = 0, 0.6
steps = 70

# シミュレーションループ
for i, angle in enumerate(angle_range):
    for j, y_offset in enumerate(y_offset_range):
        
        # 初期条件の設定（例: 2つの粒子）
        p = np.array([[40, 40], [60, 40+ y_offset ]])
        v = np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle))],
                      [np.cos(np.radians(angle+90)), np.sin(np.radians(angle+90))]])
        
        # シミュレーションの実行
        prev_p = np.copy(p)
        for step in range(steps):
            field = concentration_field(mesh, len(p), p, v, a, b, R, k1, L)
            v = update_velocity_with_gradient(p, v, field, dt, alpha, mesh, k2)
            p = update_position(p, v, dt, mesh)
            prev_p = np.copy(p)
        
        # 反射角が発生したときのオーダパラメータを取得
        order_parameter = calculate_order_parameter(v)
        order_parameters[i, j] = order_parameter

# 相図の描画
plt.figure(figsize=(8, 6))
plt.pcolormesh(angle_range, y_offset_range, order_parameters.T, cmap='jet', shading='auto')
plt.colorbar(label='Order Parameter')
plt.xlabel('Incidence Angle (degrees)')
plt.ylabel('Particle Y-Offset')
plt.title('Phase Diagram of Order Parameter')
plt.show()
