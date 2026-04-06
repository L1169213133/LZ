import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 生成数据: y = 2x + 1 + noise
n_samples = 100
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_true = 2 * X.ravel() + 1
noise = np.random.normal(0, 0.5, n_samples)
y = y_true + noise

# 添加偏置项 (x0 = 1)
X_b = np.c_[np.ones((n_samples, 1)), X]

# 定义损失函数
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    return (1/(2*m)) * np.sum((predictions - y) ** 2)

# 批量梯度下降 (GD)
def gd(X, y, lr=0.01, n_iter=200):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化为[0, 0]
    losses = []
    
    for _ in range(n_iter):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
        losses.append(compute_loss(X, y, theta))
    
    return losses

# 随机梯度下降 (SGD)
def sgd(X, y, lr=0.01, n_iter=200):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    
    for t in range(n_iter):
        # 随机选择一个样本
        idx = np.random.randint(m)
        xi, yi = X[idx:idx+1], y[idx:idx+1]
        
        # 计算梯度并更新
        gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta -= lr * gradient
        
        # 每次迭代后记录整体损失
        losses.append(compute_loss(X, y, theta))
    
    return losses[:n_iter]  # 确保长度一致

# 执行算法
losses_gd = gd(X_b, y, lr=0.01, n_iter=200)
losses_sgd = sgd(X_b, y, lr=0.01, n_iter=200)

# 绘制对比图 (全部使用拼音)
plt.figure(figsize=(12, 6))
plt.plot(losses_gd, label='Pi Liang Ti Du Xia Jiang (GD)', linewidth=2.5)
plt.plot(losses_sgd, label='Sui Ji Ti Du Xia Jiang (SGD)', alpha=0.8, linewidth=1.5)
plt.xlabel('Die Dai Ci Shu', fontsize=12)
plt.ylabel('Sun Shi Han Shu Zhi (MSE)', fontsize=12)
plt.title('GD vs SGD Shou Lian Xing Wei Dui Bi', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')  # 对数刻度更清晰
plt.tight_layout()
plt.show()

# 打印最终损失
print(f"GD Final Loss: {losses_gd[-1]:.6f}")
print(f"SGD Final Loss: {losses_sgd[-1]:.6f}")