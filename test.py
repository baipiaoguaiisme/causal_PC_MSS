from sparse_shift.causal_learn.PC import augmented_pc
import numpy as np
from causallearn.search.ConstraintBased import PC
from sparse_shift.methods import MinChange
from sparse_shift import utils
from causallearn.utils.cit import kci
from typing import Union, Optional


"""
冰淇淋销售量、溺水人数、空气温度
"""


import numpy as np
import pandas as pd


# 冰淇淋、温度、溺水人数
def generate_confounded_data(n_samples, climate_type):
    """
    根据气候类型生成模拟数据。

    因果结构:
    空气温度 -> 冰淇淋销量
    空气温度 -> 溺水人数

    P(冰淇淋销量 | 空气温度) 和 P(溺水人数 | 空气温度) 在两种气候下是*不变*的。
    P(空气温度) 是*可变*的（这代表了机制转变）。
    """

    # 1. 设置空气温度 (T)
    # 这是机制转变发生的地方
    if climate_type == 'cold':
        # 常年寒冷：温度均值较低 (例如, 10°C)
        temp_mean = 10
        temp_std = 5  # 季节性波动
    elif climate_type == 'hot':
        # 常年炎热：温度均值较高 (例如, 30°C)
        temp_mean = 30
        temp_std = 5  # 季节性波动
    else:
        raise ValueError("未知的气候类型")

    # 生成温度数据
    temperature = np.random.normal(loc=temp_mean, scale=temp_std, size=n_samples)

    # 2. 定义 *不变的* 因果机制

    # 机制 1: 温度 -> 冰淇淋销量 (I)
    # 我们使用泊松分布（Poisson）来模拟“销量”（计数数据）
    # 销量率 (lambda) 随温度指数增长
    # log(lambda_I) = a*T + b
    # 我们设定参数 a=0.07, b=3.2
    lambda_ice_cream = np.exp(0.07 * temperature + 3.2)
    ice_cream_sales = np.random.poisson(lambda_ice_cream)

    # 机制 2: 温度 -> 溺水人数 (D)
    # 同样使用泊松分布来模拟“事件发生次数”
    # 溺水率 (lambda) 也随温度指数增长，但基础率更低
    # log(lambda_D) = c*T + d
    # 我们设定参数 c=0.08, d=-0.1
    lambda_drowning = np.exp(0.08 * temperature - 0.1)
    # 确保 lambda 不为负（尽管 exp 已经保证了）
    lambda_drowning[lambda_drowning < 0] = 0
    drowning_incidents = np.random.poisson(lambda_drowning)

    # 3. 创建 DataFrame
    df = pd.DataFrame({
        'IceCreamSales': ice_cream_sales,
        'AirTemperature': temperature,

        'DrowningIncidents': drowning_incidents
    })

    return df


def simulation(plot=False):
    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 为每个国家生成 1000 天的数据
    N_SAMPLES = 1000

    # --- 生成寒冷国家的数据 (环境 1) ---
    df_cold = generate_confounded_data(N_SAMPLES, 'cold')
    # df_cold['Environment'] = 'Cold'

    # --- 生成炎热国家的数据 (环境 2) ---
    df_hot = generate_confounded_data(N_SAMPLES, 'hot')
    # df_hot['Environment'] = 'Hot'

    df_all = pd.concat([df_cold, df_hot])

    feature_names = df_all.columns.tolist()
    # NetworkX 默认节点为 0, 1, 2...，因此创建映射: {0: 'IceCreamSales', 1: 'AirTemperature', ...}
    node_labels = {i: name for i, name in enumerate(feature_names)}

    df_all = df_all.to_numpy()
    print(df_all.shape)

    # mec
    mec = PC.pc(df_all, indep_test=kci, node_names=feature_names,mvpc=False)
    print()
    print('PC算法：')
    print(mec.G.graph)
    print('-' * 100)

    # 增强mec
    aug_mec = augmented_pc(df_all, indep_test=kci, cg=mec)

    print('加强PC算法：')
    print(aug_mec.G.graph)
    print('-' * 100)

    # # MSS
    mss = MinChange(mec.G.graph)
    mss.add_environment(df_cold.to_numpy())
    mss.add_environment(df_hot.to_numpy())

    print('MSS:')
    print('CPDAG:')
    print(mss.get_min_cpdag())
    print('-' * 100)
    print('DAG')
    print(mss.get_min_dags())
    np.save('min_dags.npy',mss.get_min_dags())
    print(np.load('min_dags.npy'))

    if plot:
        plot_graph(mss.get_min_cpdag(), node_labels,save=True)



def plot_graph(g_numpy: np.ndarray, node_labels: Optional[Dict] = None,save=False,save_file='causal graph.png'):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    # --- 添加 Matplotlib 中文支持配置 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    # ----------------- 1. 定义邻接矩阵 -----------------
    # 4 个节点, 边: 0->1, 1->2, 2->0, 3->3
    adjacency_matrix = np.array(g_numpy)

    # ----------------- 2. 转换为 NetworkX 有向图 -----------------
    # 确保使用 create_using=nx.DiGraph 创建有向图
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    # 如果您的节点需要自定义标签（例如 'X1', 'X2', 'X3', 'X4'）
    # 您可以创建一个映射，然后传入 draw_networkx 的 labels 参数
    # node_labels = {i: f'X{i+1}' for i in G.nodes}

    # ----------------- 3. 可视化图 -----------------
    plt.figure(figsize=(16,12))

    # 使用 spring_layout 布局
    pos = nx.spring_layout(G, seed=42)  # 设置 seed 以获得一致的布局

    # 绘制图
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        labels=node_labels,  # 启用自定义标签时使用
        node_size=2500,
        node_color="lightgreen",
        font_size=18,
        font_weight="bold",
        edge_color="darkgray",
        arrows=True,
        arrowstyle="->",
        arrowsize=25,
        width=2
    )

    plt.title("有向图可视化", fontsize=20)
    plt.axis('off')
    # plt.show()
    if save:
        plt.savefig(save_file)


# --- 主程序 ---
if __name__ == "__main__":
    simulation(True)
