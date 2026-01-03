import copy
import matplotlib.pyplot as plt
import numpy as np

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i], w)+ b)- y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err* x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def compute_cost(x, y, w, b):
    cost = 0.
    m = x.shape[0]
    for i in range(m):
        f_wb_i = np.dot(x[i], w)+ b
        cost = cost + (f_wb_i-y[i])**2
    cost = cost/(2* m)
    return cost

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    w_history = [w.copy()]
    b_history = [b]

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w = w - alpha* dj_dw
        b = b - alpha* dj_db
        tmp_cost = cost_function(x, y, w, b)

        j_history.append(tmp_cost)
        w_history.append(w)
        b_history.append(b)

    return w, b, j_history, np.array(w_history), np.array(b_history)

def plot_gradient_descent_results(cost_history, w_history, b_history):
    """
    代价函数随迭代的变化
    """
    plt.figure(figsize=(14, 10))

    # plt.subplot(1,1,1)
    plt.plot(range(len(cost_history)), cost_history, 'b-o', linewidth=2)
    plt.yscale('log')
    plt.title('Cost Change with Iters', fontsize=14)
    plt.xlabel('number of iters', fontsize=12)
    plt.ylabel('cost(log)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)      # 启用虚线网格，透明度70%

    w_str = ','.join([f'{w:.6f}'for w in w_history[0]])

    # add the note of the first and the end
    plt.annotate(f'first cost: {cost_history[0]:.2e}\nw: [{w_str}], \nb: {b_history[0]:.6f}',
                 xy=(0, cost_history[0]),
                 xytext=(10, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'))

    w_str = ','.join([f'{w:.6f}'for w in w_history[-1]])
    plt.annotate(f'end cost: {cost_history[-1]:.2e}'+f'\nw: [{w_str}]'+f"\nb: {b_history[-1]:.6f}" ,
                 xy=(len(cost_history) - 1, cost_history[-1]),
                 xytext=(-100, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.savefig('C1_W2_Lab02_GDWithMultipleVariables.png')
    plt.show()


if __name__ == '__main__':
    x_train = np.array([[2104, 5, 1, 45],
                       [1416, 3, 2, 40],
                       [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_target = 785.1811367994083
    w_target = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


    b_init = 0.
    w_init = np.zeros_like(w_target)

    init_cost = compute_cost(x_train, y_train, w_init, b_init)
    print(f"初始代价: {init_cost:.6f}")

    alpha = 5e-8
    num_iters = 100

    w_final, b_final, cost_history, w_history, b_history = gradient_descent(
        x_train, y_train, w_init, b_init,
        compute_cost,compute_gradient,
        alpha=alpha,
        num_iters=num_iters
    )

    print(f"理想参数: w = {w_target}, b = {b_target:.6f}")
    print(f"最终参数: w = {w_history[-1]}, b = {b_history[-1]:.6f}")
    print(f"最终代价: {cost_history[-1]:.6f}")

    plot_gradient_descent_results(cost_history, w_history, b_history)

