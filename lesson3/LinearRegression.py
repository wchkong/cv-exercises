import numpy as np
import random


def inference(w, b, x_list):
    pred_y_list = w * np.array(x_list) + b
    return pred_y_list


def eval_loss(w, b, x_list, gt_y_list):
    loss = 0.5 * np.power((w * np.array(x_list) + b - gt_y_list), 2)
    avg_loss = np.mean(loss)
    return avg_loss


def gradient(pred_y_list, gt_y_list, x_list):
    diff = pred_y_list - gt_y_list
    dw = np.mean(diff * x_list)
    db = np.mean(diff)
    return dw, db


def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    pred_y_list = inference(w, b, batch_x_list)
    avg_dw, avg_db = gradient(pred_y_list, batch_gt_y_list, batch_x_list)
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def train(x_list, y_list, batch_size, lr, max_iter):
    w = 0.0
    b = 0.0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_list, y_list)))


def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b


def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)


if __name__ == '__main__':
    run()
