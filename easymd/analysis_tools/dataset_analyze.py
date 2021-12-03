import argparse
import numpy as np
import torch
import os.path as osp
import matplotlib.pyplot as plt

#from mmdet import cv_core

from mmcv import Config
import mmcv
from mmdet.datasets.builder import build_dataset, build_dataloader
class Kmean(object):
    def __init__(self, cluster_number, number_iter=1, name='iou'):
        self.cluster_number = cluster_number
        self.number_iter = number_iter
        self.name = name

    def _get_distance_measure(self, name='iou'):
        if name == 'iou':
            return self._calc_iou
        else:
            raise NotImplementedError('暂时没有实现')

    def _calc_iou(self, boxes_nx2, clusters_kx2):
        """
        calculate the iou between bboxes and clusters
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
            clusters_kx2(np.ndarray): clusters_kx2's width and height
        return:
            iou_nxk(np.ndarray): iou between bboxes and clusters
        """
        n = boxes_nx2.shape[0]
        k = self.cluster_number

        box_area = boxes_nx2[:, 0] * boxes_nx2[:, 1]  # 相当于左上角全部移动到0,0点，进行iou计算
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters_kx2[:, 0] * clusters_kx2[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes_nx2[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters_kx2[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes_nx2[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters_kx2[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        iou_nxk = inter_area / (box_area + cluster_area - inter_area)
        return iou_nxk

    def _calc_average_measure(self, boxes_nx2, clusters_kx2):
        """
        calculate the mean iou between bboxes and clusters
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
            clusters_kx2(np.ndarray): clusters_kx2's width and height
        return:
            mean_iou(np.ndarray): mean iou between boxes and their corresponding clusters
        """
        _distance_measure_fun = self._get_distance_measure(self.name)
        accuracy = np.mean([np.max(_distance_measure_fun(boxes_nx2, clusters_kx2), axis=1)])
        return accuracy

    def _kmeans(self, boxes_nx2):
        """
        cacluate the clusters by kmeans
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
        would use:
            cluster_number
        would call:
            _calc_iou()
        return:
            clusters(np.ndarray): the anchors for yolo
        """
        k = self.cluster_number
        box_number = boxes_nx2.shape[0]
        last_nearest = np.zeros((box_number,))
        clusters = boxes_nx2[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        _distance_measure_fun = self._get_distance_measure(self.name)
        while True:
            # 距离度量准则是1-iou，iou越大则越近
            distances = 1 - _distance_measure_fun(boxes_nx2, clusters)  # 输出维度 N,k

            current_nearest = np.argmin(distances, axis=1)  # 找出某个点离所有中心最近的索引
            if (last_nearest == current_nearest).all():  # 收敛
                break  # clusters won't change
            for cluster in range(k):  # 更新聚类中心
                if len(boxes_nx2[current_nearest == cluster]) == 0:
                    clusters[cluster] = boxes_nx2[np.random.choice(
                        box_number, 1, replace=False)]
                else:
                    clusters[cluster] = np.median(  # update clusters
                        boxes_nx2[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def clusters(self, wh_data_nx2):
        total_acc = -1
        total_result = []
        for _ in range(self.number_iter):
            result = self._kmeans(wh_data_nx2)  # TODO ga+kmean
            anchor_area = result[:, 0] * result[:, 1]
            area_index = np.argsort(anchor_area)
            result = result[area_index]
            acc = self._calc_average_measure(wh_data_nx2, result) * 100
            if acc > total_acc:
                total_acc = acc
                total_result = result

        # print("K anchors:\n {}".format(total_result.astype(np.int32)))
        print("Accuracy: {:.2f}%".format(total_acc))
        return total_result.astype(np.int32).tolist()

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(  # datalayer重复次数，由于数据有增强，故在数据量小的时候可以设置重复次数，统计更加准确
        '--repeat_count',
        type=int,
        default=1,
        help='datalayer repeat count')
    parser.add_argument(  # dataloader参数
        '--samples_per_gpu',
        type=int,
        default=32,
        help='batch size')
    parser.add_argument(  # dataloader参数
        '--workers_per_gpu',
        type=int,
        default=16,
        help='worker num')
    parser.add_argument(  # 统计得到的wh数据保存名称
        '--out_path',
        type=str,
        default='wh_data.npy',
        help='save wh data npy path')
    parser.add_argument(  # 当本地有缓存时候，是否使用，而不重新经过datalayer,节省时间
        '--use_local',
        type=bool,
        default=True,
        help='is use save npy file')
    args = parser.parse_args()
    return args


def collect_wh_data(cfg, args, stop_count):
    # stop_count 防止数据太大，要很久才能跑完

    dataset = build_dataset(cfg.data.train)  # 这样才能考虑到数据增强带来的图片比例改变
    dataloader = build_dataloader(dataset, args.samples_per_gpu, args.workers_per_gpu)
    print('----开始遍历数据集----')
    wh_all = []
    for count in range(args.repeat_count):
        progress_bar = mmcv.ProgressBar(len(dataloader))
        for i, data_batch in enumerate(dataloader):
            if i > stop_count:
                break
            gt_bboxes = data_batch['gt_bboxes'].data[0]
            gt_bboxes = torch.cat(gt_bboxes, dim=0).numpy()
            if len(gt_bboxes) == 0:
                continue
            w = (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            wh = np.stack((w, h), axis=1)
            wh_all.append(wh)
            progress_bar.update()
    wh_all = np.concatenate(wh_all, axis=0)
    print(wh_all.shape)
    return wh_all


def select_data(cfg, args, stop_count=100):
    use_local = args.use_local
    out_path = args.out_path
    if not use_local or not osp.isfile(out_path):
        print('--------重新获取数据---------')
        wh_all_data = collect_wh_data(cfg, args, stop_count)
        np.save(out_path, wh_all_data)
        print('---------保存缓存文件--------')
    else:
        # 直接读取
        print('---------从缓存文件中读取---------')
        wh_all_data = np.load(out_path)
    return wh_all_data


def statistics_hw_ratio(wh_all):
    print('----------统计宽高分布---------')
    # 部分参考：https://zhuanlan.zhihu.com/p/108885033
    hw_ratio = wh_all[:, 1] / wh_all[:, 0]  # anchor里面的ratio就是h/w比例

    # 分成两部分单独统计
    hw_ratio_larger = hw_ratio[hw_ratio >= 1].astype(np.int)  # 会损失些精度
    hw_ratio_larger_uq = np.unique(hw_ratio_larger)
    box_hw_larger_count = [np.count_nonzero(hw_ratio_larger == i) for i in hw_ratio_larger_uq]

    plt.subplot(2, 1, 1)
    plt.title('hw_ratio>=1')
    plt.xlabel('hw_ratio')
    plt.ylabel('num')
    plt.bar(hw_ratio_larger_uq, box_hw_larger_count, 0.1)  # 0-20之间
    # # wh_df = pd.DataFrame(box_hw_larger_count, index=hw_ratio_larger_uq, columns=['hw_ratio>=1'])
    # # wh_df.plot(kind='bar', color="#55aacc")

    hw_ratio_small = hw_ratio[hw_ratio < 1].round(1)
    hw_ratio_small_uq = np.unique(hw_ratio_small)
    box_hw_small_count = [np.count_nonzero(hw_ratio_small == i) for i in hw_ratio_small_uq]

    plt.subplot(2, 1, 2)
    plt.title('hw_ratio<1')
    plt.xlabel('hw_ratio')
    plt.ylabel('num')
    plt.bar(hw_ratio_small_uq, box_hw_small_count, 0.05)  # 0-1之间

    plt.show()

    hw_ratio = np.concatenate((hw_ratio_small, hw_ratio_larger), axis=0).round(1)
    hw_ratio_uq = np.unique(hw_ratio).tolist()
    box_hw_count = [np.count_nonzero(hw_ratio == i) for i in hw_ratio_uq]

    print('按照num数从大到小排序输出')
    data = sorted(zip(hw_ratio_uq, box_hw_count), key=lambda x: x[1], reverse=True)
    hw_ratio_uq, box_hw_count = zip(*data)
    print('hw_ratio', hw_ratio_uq)
    print('num', box_hw_count)


def statistics_hw_scale(wh_data):
    print('----------统计wh尺度分布---------')
    # plt.scatter(wh_data[:,0],wh_data[:,1])
    plt.subplot(2, 1, 1)
    plt.xlabel('w_scale')
    plt.ylabel('num')
    plt.hist(wh_data[:, 0], bins=1000)
    plt.subplot(2, 1, 2)
    plt.xlabel('h_scale')
    plt.ylabel('num')
    plt.hist(wh_data[:, 1], bins=1000)
    plt.show()


def calc_kmean(wh_data):
    print('----------统计anchor分布---------')
    cluster_number = 9  # anchor个数
    kmean_clz = Kmean(cluster_number)
    anchor_nx2 = kmean_clz.clusters(wh_data)
    print("K anchors:\n {}".format(anchor_nx2))


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    wh_data = select_data(cfg, args)

    statistics_hw_ratio(wh_data)
    statistics_hw_scale(wh_data)
    calc_kmean(wh_data)



