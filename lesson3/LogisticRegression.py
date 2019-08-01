import numpy as np
import random


def inference(w, x):
    pred_y = 1 / (1 + np.exp(0 - w * x))
    return pred_y


def eval_loss(w, x_list, gt_y_list):
    avg_loss = 0.0
    # cost(hθ(x),y)=−yi * log(hθ(x)) − (1 − yi) * log(1 − hθ(x))
    for i in range(len(x_list)):
        pred_y = inference(w, x_list[i])
        avg_loss += -1 * gt_y_list[i] * np.log(pred_y) - (1 - gt_y_list[i]) * np.log(1 - pred_y)
    avg_loss /= len(gt_y_list)
    return avg_loss


def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    return dw


def cal_step_gradient(batch_x_list, batch_gt_y_list, w, lr):
    avg_dw = 0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pred_y = inference(w, batch_x_list[i])
        dw = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
    avg_dw /= batch_size
    w -= lr * avg_dw
    return w


def train(x_list, y_list, batch_size, lr, max_iter):
    w = 0.0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [y_list[j] for j in batch_idxs]
        w = cal_step_gradient(batch_x, batch_y, w, lr)
        loss = eval_loss(w, x_list, y_list)
        if loss <= 0.05:
            break
        print('w:{0}'.format(w))
        print('loss is {0}'.format(loss))


def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(-100, 100) * random.random()
        y = np.power((1 + np.exp(0 - w * x)), -1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w


def run():
    x_list, y_list, w = gen_sample_data()
    lr = 0.001
    max_iter = 2000
    train(x_list, y_list, 50, lr, max_iter)


if __name__ == '__main__':
    run()
