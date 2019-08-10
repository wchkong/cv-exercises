import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 选择初始点
def get_centroids(df, k):
    centroids = {
        i: [0, 0]
        for i in range(k)
    }
    # 1.随机选择一个点作为一个中心点
    c1_idx = np.random.randint(0, len(df['x']))
    centroids[0] = [df['x'][c1_idx], df['y'][c1_idx]]
    # 2.计算各点到当前中心点的距离
    for i in range(k - 1):
        center = centroids[i]
        # 2.1 计算距离的平方
        df['D_X_2_{}'.format(i)] = ((df['x'] - center[0]) ** 2
                                    + (df['y'] - center[1]) ** 2)

        sum_dx2 = sum(df['D_X_2_{}'.format(i)])
        # 2.2 计算每个的的概率
        df['P_X_{}'.format(i)] = (df['D_X_2_{}'.format(i)] / sum_dx2)

        # 2.3 计算向前加和的概率值
        df['SUM_P_{}'.format(i)] = [sum(df['P_X_{}'.format(i)][0: j + 1]) for j in range(len(df))]

        # 2.4 随机求得新的center, 添加到结果列表中
        random_p = np.random.rand()
        array = np.where(df['SUM_P_{}'.format(i)] >= random_p)
        new_idx = array[0][0]
        centroids[i + 1] = [df['x'][new_idx], df['y'][new_idx]]

    return centroids


def assignment(df, centroids, colmap):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


def update(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids


def main():
    # step 0.0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3
    centroids = get_centroids(df, k)

    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        # key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            print('THE END!')
            break


if __name__ == '__main__':
    main()
