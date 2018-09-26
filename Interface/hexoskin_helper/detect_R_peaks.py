import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_store
import scipy.signal as signal
import math
import fastdtw

measures = {}

def rollmean(dataset, hrw, fs):
    mov_avg = dataset.heart.rolling(int(hrw * fs)).mean()
    avg_hr = np.mean(dataset.heart)

    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x * 1.06 for x in mov_avg]
    measures['heart_rolling_mean'] = mov_avg

def detect_peaks(dataset):
    window = []
    window_indices = []
    peaklist = []

    listpos = 0

    for datapoint in dataset.heart:
        rollingmean = measures['heart_rolling_mean'][listpos]
        if(datapoint < rollingmean) and (len(window) < 1):
            listpos += 1
        elif datapoint > rollingmean:
            window.append(datapoint)
            window_indices.append(listpos)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            window_indices = []
            listpos += 1

    measures['xbeat'] = peaklist
    measures['ybeat'] = [dataset.heart[x] for x in peaklist]

def calculate_RR_list(fs):
    RR_list = []
    xbeat = measures['xbeat'][1:]
    count = 0
    while(count < len(xbeat) - 1):
        rr_interval = xbeat[count + 1] - xbeat[count]
        seconds_dist = rr_interval / fs
        RR_list.append(seconds_dist)
        count += 1
    measures['RR_list'] = RR_list

def calculate_P_and_T_window(rr_interval):

    pwindow = 0.08 + (0.2 * rr_interval) + 0.1
    twindow = 1.5 * 0.42 * np.sqrt(rr_interval) - 0.08

    return pwindow, twindow

def get_segments(pwindow, twindow, beatposition, dataset, fs):
    pr_seg_samples = int(pwindow * fs)
    rt_seg_samples = int(twindow * fs)

    pr_seg = dataset.heart[beatposition - pr_seg_samples : beatposition + 1]
    rt_seg = dataset.heart[beatposition : beatposition + rt_seg_samples + 1]

    return pr_seg, rt_seg

def find_alignment(rr_interval_seg1, rr_interval_seg2, xbeat1, xbeat2, dataset, fs):
    pwindow_seg1, twindow_seg1 = calculate_P_and_T_window(rr_interval_seg1)
    pwindow_seg2, twindow_seg2 = calculate_P_and_T_window(rr_interval_seg2)

    pr_seg1, rt_seg1 = get_segments(pwindow_seg1, twindow_seg1, xbeat1, dataset, fs)
    pr_seg2, rt_seg2 = get_segments(pwindow_seg2, twindow_seg2, xbeat2, dataset, fs)

    w = 0.1 * fs

    distance1, path1 = fastdtw.fastdtw(pr_seg1, pr_seg2, radius=w)
    distance2, path2 = fastdtw.fastdtw(rt_seg1, rt_seg2, radius=w)
    return np.add(distance1, distance2)

def initialize_centroids(V, K):
    centroids = V.copy()
    np.random.shuffle(centroids)
    return centroids[:K]

def move_centroids(V, K, classes):
    return np.array([V[np.where(classes == k)].mean(axis=0) for k in range(K)])

def find_classes(V, centroids):
    classes = np.zeros((V.shape[0], centroids.shape[0]), dtype=int)
    class_final = np.zeros((V.shape[0], 1), dtype=int)
    for i in range(V.shape[0]):
        for j in range(centroids.shape[0]):
            classes[i, j] = np.sqrt(np.sum(np.square(V[i] - centroids[j])))
        class_final[i] = np.argmin(classes[i])
    return class_final

def k_means(V, K, num_iters):
    centroids = initialize_centroids(V, K)
    classes = np.array((V.shape[0], 1), dtype=int)
    for i in range(num_iters):
        classes = find_classes(V, centroids)
        centroids = move_centroids(V, K, classes)
    return centroids

def find_max_cluster_index(V, centroids, K):
    classes = find_classes(V, centroids)
    return np.argmax(np.array([np.sum(classes == k) for k in range(K)]))

def calc_distance(V, centroids, cluster_index):
    classes = find_classes(V, centroids)
    internal_distance = np.array([], dtype=float)
    external_distance = np.array([], dtype=float)
    for i in range(len(classes)):
        int_dist = np.array([], dtype=float)
        ext_dist = np.array([], dtype=float)
        if classes[i] == cluster_index:
            for j in range(len(classes)):
                if classes[j] == cluster_index:
                    np.append(int_dist, V[i, j])
                else:
                    np.append(ext_dist, V[i, j])
            np.append(internal_distance, np.mean(int_dist))
            np.append(external_distance, np.mean(ext_dist))
    return internal_distance, external_distance

def find_normal_beat_index(dataset, fs):
    xbeat = measures['xbeat'][2:]
    RR_list = measures['RR_list']
    V = np.zeros((len(xbeat), len(xbeat)), dtype=float)
    num_clusters = 2
    for i in range(len(xbeat)):
        for j in range(len(xbeat)):
            V[i, j] = find_alignment(RR_list[i], RR_list[j], xbeat[i], xbeat[j], dataset, fs)
    final_centroids = k_means(V, K=num_clusters, num_iters=50)
    classes = find_classes(V, final_centroids)
    max_cluster_index = find_max_cluster_index(V, final_centroids, num_clusters)

    internal_distance, external_distance = calc_distance(V, final_centroids, max_cluster_index)
    print(internal_distance, external_distance)

def plot(dataset):
    plt.title("Detected peaks in signal")
    plt.plot(dataset.heart, alpha=0.5, color='blue', label='raw signal')
    plt.plot(measures['heart_rolling_mean'], color='green', label='moving average')
    plt.scatter(measures['xbeat'], measures['ybeat'], color='red')
    plt.ylim(1200, 1600)
    plt.show()

def process(hrw, fs):
    dataset = data_store.get_record_data(155604)
    rollmean(dataset, hrw, fs)
    detect_peaks(dataset)
    calculate_RR_list(fs)
    find_normal_beat_index(dataset, fs)
    plot(dataset)

if __name__ == '__main__':
    process(0.292, 256)
