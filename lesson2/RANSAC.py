import random

#random sample consensus
"""
    参考：https://blog.csdn.net/vict_wang/article/details/81027730
    RANSAC算法伪代码：
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回None,如果未找到）

    #遍历次数
    iterations = 0
    #最优解记录
    bestfit = None
    #最优解误差
    besterr = something really large(取一极大值)
    while iterations < k:
        points_maybe_in_line = data中随机取n个点作为局内点
        maybe_model = 根据points_maybe_in_line进行拟合生成的模型
        #存放局外点中符合模型的点
        points_also_in_line = emptyset
        for point in data but not in points_maybe_in_line:
            if error(point in maybe_model) < t:
                points_also_in_line.add(point)

        if (len(points_also_in_line) > d):
            bettermodel = 利用所有的points_maybe_in_line 和 points_maybe_in_line 重新生成更好的模型
            thiserr = 所有的points_maybe_in_line 和 points_maybe_in_line样本点的误差度量
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
        iterations += 1
    return bestfit

    """
def ransacMatching(A, B):
    # A & B: List of List  = [[1,2],[2,3]] ?

    A_size = len(A)
    B_size = len(B)
    A_inline = random.sample(A, 4)
    B_inline = random.sample(B, 4)
    #计算4个点的模型

    return


def main():
    pass


if __name__ == "__main__":
    main()
