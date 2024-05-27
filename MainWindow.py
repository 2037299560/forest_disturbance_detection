from PyQt5 import QtWidgets
from PyQt5.QtChart import QLineSeries, QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QFileDialog

from datetime import datetime
from osgeo import gdal
from osgeo import ogr
import pandas as pd
import numpy as np
import torch
import time
import os

from algorithms import train_IRMSTransformer
from algorithms.models.IRMSTransformer import IRMSTransformer
from forest_disturbance import Ui_mainWindow

class MainWindow(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # 定义变量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 控件初始化
        self.label_5.setText("")
        # 绑定事件
        self.events_bind()

        self.load_time_series_in_model_analysis()

    def events_bind(self):
        ###
        ### deep_learning 模块
        ###
        self.pushButton_10.clicked.connect(lambda: self.load_file(self.lineEdit_9))
        self.pushButton_14.clicked.connect(lambda: self.save_file(self.lineEdit_12))
        self.pushButton_11.clicked.connect(lambda: self.load_file(self.lineEdit_10))
        self.pushButton_13.clicked.connect(lambda: self.load_file(self.lineEdit_11))
        self.pushButton_15.clicked.connect(lambda: self.save_file(self.lineEdit_13))

        self.pushButton_12.clicked.connect(self.supervised_train)

        ###
        ### 数据导入及预处理模块
        ###
        # 数据加载
        self.pushButton.clicked.connect(lambda: self.load_dir(self.lineEdit))
        self.pushButton_2.clicked.connect(lambda: self.load_dir(self.lineEdit_2))
        self.pushButton_3.clicked.connect(lambda: self.load_file(self.lineEdit_3))
        self.pushButton_4.clicked.connect(lambda: self.load_file(self.lineEdit_4))
        self.pushButton_5.clicked.connect(lambda: self.load_file(self.lineEdit_5))
        # 数据预处理保存路径
        self.pushButton_6.clicked.connect(lambda: self.load_dir(self.lineEdit_6))
        # 样本数据栅格化
        self.pushButton_8.clicked.connect(self.train_test_samples_rasterize)

        ###
        ### 模型分析
        ###
        #数据加载
        self.pushButton_25.clicked.connect(lambda: self.load_file(self.lineEdit_19))
        self.pushButton_26.clicked.connect(lambda: self.load_file(self.lineEdit_18))
        self.pushButton_24.clicked.connect(lambda: self.load_file(self.lineEdit_17))
        # 模型加载
        self.pushButton_28.clicked.connect(lambda: self.load_file(self.lineEdit_20))

        ###
        ### 干扰检测与结果可视化
        ###
        # 数据加载
        self.pushButton_17.clicked.connect(lambda: self.load_file(self.lineEdit_14))
        self.pushButton_18.clicked.connect(lambda: self.load_file(self.lineEdit_8))
        self.pushButton_9.clicked.connect(lambda: self.load_file(self.lineEdit_7))
        # 模型加载
        self.pushButton_20.clicked.connect(lambda: self.load_file(self.lineEdit_15))
        # 结果保存
        self.pushButton_23.clicked.connect(lambda: self.save_file(self.lineEdit_16))

    def load_file(self, display_object):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if file_name:
            print(file_name)
            # 将读取的文件名写入到lineEdit_9
            display_object.setText(file_name)

    def load_dir(self, display_object):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        dir_name = QFileDialog.getExistingDirectory(None, "QFileDialog.getExistingDirectory()", options=options)
        if dir_name:
            print(dir_name)
            # 将读取的文件名写入到lineEdit_9
            display_object.setText(dir_name)
    # 保存文件
    def save_file(self, display_object):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(None, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if file_name:
            print(file_name)
            # 将读取的文件名写入到lineEdit_9
            display_object.setText(file_name)

    '''
    数据导入模块
    '''
    @staticmethod
    def shp2raster(shp_file, Sentinel2_reference_file, output_path):
        # 输出栅格的文件路径
        tmp1 = os.path.join(output_path, "data_import")
        if not os.path.exists(tmp1):
            os.makedirs(tmp1)
        out_raster_file = os.path.join(tmp1, os.path.basename(shp_file).split(".")[0] + ".tif")

        # 使用GDAL OGR库读取shp文件
        dataSource = ogr.Open(shp_file)
        layer = dataSource.GetLayer()

        # 读取参考栅格
        raster = gdal.Open(os.path.join(Sentinel2_reference_file, "B03.tif"))
        projection = raster.GetProjection()
        transform = raster.GetGeoTransform()
        cols = raster.RasterXSize
        rows = raster.RasterYSize


        # 创建输出栅格图层
        target_ds = gdal.GetDriverByName('GTiff').Create(out_raster_file, cols, rows, 1, gdal.GDT_Int32)
        target_ds.SetGeoTransform(transform) # 设置地理变换信息
        target_ds.SetProjection(projection) # 设置投影信息
        # 设置栅格图层的无效值
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.FlushCache()

        # 将shp文件转换为栅格数据，并将burn_values设置为1（即所有矢量要素都被标记为1）
        # options=["ALL_TOUCHED=TRUE"]将会将所有覆盖到的像元的值都设为burn_value，即使像元只有一部分被覆盖

        gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=type_code"])

        # 保存输出栅格图层
        target_ds.FlushCache()
        target_ds = None

    # 矢量样本栅格化
    def train_test_samples_rasterize(self):
        train_samples_file = self.lineEdit_4.text()
        test_samples_file = self.lineEdit_5.text()
        Sentinel2_reference_file = self.lineEdit_2.text()
        output_path = self.lineEdit_6.text()
        print(f"train_samples_file: {train_samples_file}\n"
                                f"test_samples_file: {test_samples_file}\n"
                                f"Sentinel2_reference_file: {Sentinel2_reference_file}\n"
                                f"output_path: {output_path}\n")
        self.shp2raster(train_samples_file, Sentinel2_reference_file, output_path)
        self.shp2raster(test_samples_file, Sentinel2_reference_file, output_path)

        # 输出结果至label_7
        self.label_7.setText(f"train_samples_file: {train_samples_file}\n"
                                f"test_samples_file: {test_samples_file}\n"
                                f"Sentinel2_reference_file: {Sentinel2_reference_file}\n"
                                f"output_path: {output_path}\n")
        self.label_7.setText("训练样本和测试样本栅格化完成...")




    # 模型监督训练
    def supervised_train(self):
        train_data_dir = r"D:\Work_Space\Forest_Disturbance_Puer_src\data\supervised\original_bands\train"
        val_data_dir = r"D:\Work_Space\Forest_Disturbance_Puer_src\data\supervised\original_bands\test"
        test_data_dir = r"D:\Work_Space\Forest_Disturbance_Puer_src\data\supervised\original_bands\test"
        model_save_path = r"D:\Work_Space\Software_Forest_Disturbance\algorithms\sources\1.pth"

        train_IRMSTransformer.main(train_data_dir, val_data_dir, test_data_dir, model_save_path, output_widget=self.label_5)

    def load_time_series_in_model_analysis(self):
        # 创建一个柱状集并添加数据
        set0 = QBarSet("Bar1")
        set0 << 1 << 2 << 3 << 4 << 5

        # 创建一个柱状序列并添加柱状集
        series = QBarSeries()
        series.append(set0)

        # 创建一个图表，添加序列
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Simple Bar Chart Example")

        # 创建一个数值轴，设置其最小值、最大值和步长
        axisX = QValueAxis()
        axisX.setRange(0, 5)
        axisX.setTickInterval(1)

        # 将数值轴添加到图表中
        chart.setAxisX(axisX, series)

        # 创建一个图表视图，设置抗锯齿渲染并设置图表
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.addWidget(chart_view)

        self.widget_2.setLayout(layout)
        # # 读取Sentinel-2多波段数据
        # print("加载Sentinel-2数据...")
        # s2_data = np.load("analysis/不规则时间序列分析/data/s2_data.npy")
        #
        #
        # series = QLineSeries()
        # series.append(0, 1)
        # series.append(1, 3)
        # series.append(2, 7)
        # series.append(3, 6)
        # series.append(4, 4)
        #
        # chart = QChart()
        # chart.addSeries(series)
        # chart.setTitle("")
        # chart.createDefaultAxes()
        #
        # # 创建图表视图并设置图表
        # chart_view = QChartView(chart, self.widget)
        # chart_view.setRenderHint(QPainter.Antialiasing)
        # layout = QtWidgets.QVBoxLayout(self.widget)
        # layout.addWidget(chart_view)
        #
        # self.widget.setLayout(layout)
        # # 将图表视图添加到UI中的布局中
        # # self.widget.setCentralWidget(chart_view)

    def cal_grad_cam(self, feature_map, feature_grad):
        '''
        feature_map: (ts_length, samples, channels)
        feature_grad: (ts_length, samples, channels)
        '''
        # 根据梯度计算特征图权重
        weights_gradients = np.mean(feature_grad, axis=0)  # (samples, channels)
        # 根据权重计算特征图加权和
        for i in range(feature_map.shape[2]):
            feature_map[:, :, i] *= np.tile(np.expand_dims(weights_gradients[:, i], axis=0), (feature_map.shape[0], 1))
        # 按通道求和
        cam = np.mean(feature_map, axis=2)  # (ts_length, samples)
        # 使用ReLU函数去除负值
        cam = F.relu(torch.from_numpy(cam)).numpy()
        # 所有样本求平均
        cam = np.mean(cam, axis=1)  # (ts_length, )
        # 归一化
        cam /= np.max(cam)
        return cam



    def caculate_cam_in_model_analysis(self):
        # 读取Sentinel-1多波段数据
        print("加载Sentinel-1数据...")
        s1_input = np.load(f"analysis/不规则时间序列分析/data/s1_data.npy")
        # 读取Sentinel-2多波段数据
        print("加载Sentinel-2数据...")
        s2_input = np.load(f"analysis/不规则时间序列分析/data/s2_data.npy")
        # 加载doy
        print("加载DOY...")
        doy = np.load(f"analysis/不规则时间序列分析/data/doy.npy")
        # 将数据转换为tensor
        s1_data = torch.from_numpy(s1_input).float().to(self.device).permute(2, 1, 0)  # (148, 12, 2)
        s2_data = torch.from_numpy(s2_input).float().to(self.device).permute(2, 1, 0)  # (148, 100, 11)

        ### 计算注意力图
        print("加载模型...")
        model = IRMSTransformer(
            S1_encoder_config={
                "in_channels": 2,
                "hidden_dim": 64,
                "seq_length": 12,
                "d_model": 64,
                "n_head": 4,
                "d_feedforward": 256,
                "n_layers": 2,
                "dropout": 0.4
            },
            S2_encoder_config={
                "in_channels": 10,
                "seq_length": 100,
                "hidden_dim": 64,
                "time_channels": 1,
                "d_model": 64,
                "n_head": 4,
                "d_feedforward": 256,
                "n_layers": 2,
                "dropout": 0.4
            },
            fusion_block_config={
                "d_model": 128,  # 由于两个特征进行了拼接，所以这里的d_model是两个特征的d_model之和
                "n_head": 4,
                "d_feedforward": 256,
                "n_layers": 2,
                "dropout": 0.4,
                "seq_length": 100
            },
            num_classes=3
        ).to(self.device)
        model.load("checkpoints/supervised_IRMSTransformer/irmstransformer_disturbance_type_supervised_231004.pth")
        model.eval()

        # 编写钩子函数获取输出层之前的特征向量
        fusion_feature_maps = []
        fusion_grads = []
        def fusion_hook_feature(module, input, output):
            # print("s2_hook_feature running")
            fusion_feature_maps.append(output.data.cpu().numpy())
        handle_fusion = model.feature_fusion.register_forward_hook(fusion_hook_feature)
        def fusion_backward_hook(module, grad_in, grad_out):
            fusion_grads.append(grad_out[0].detach().cpu().numpy())
        handle_fusion_grad = model.feature_fusion.register_full_backward_hook(fusion_backward_hook)

        for idx in range(s1_data.shape[0]):
            s1_in = s1_input[idx, :, :].unsqueeze(0)
            s2_in = s2_input[idx, :, :].unsqueeze(0)
            pred = model(S1=s1_in, S2=s2_in)
            model.zero_grad()
            maxpred = pred.exp().max()
            # maxpred = pred.exp().squeeze()[0]
            maxpred.backward()
        fusion_feature_map = np.concatenate(fusion_feature_maps, axis=1)  # (100, 148, 128)
        fusion_grad_feature = np.concatenate(fusion_grads, axis=1)  # (100, 148, 128)

        # 保存梯度
        np.save(f"analysis/不规则时间序列分析/data/fusion_feature_map_IRMST.npy", fusion_feature_map)
        np.save(f"analysis/不规则时间序列分析/data/fusion_feature_grad_IRMST.npy", fusion_grad_feature)

        print(f"梯度保存完成...")

        # 读取梯度，绘制柱状图
        fusion_feature_map = np.load(f"analysis/不规则时间序列分析/data/fusion_feature_map_IRMST.npy")
        fusion_grad_feature = np.load(f"analysis/不规则时间序列分析/data/fusion_feature_grad_IRMST.npy")
        # 计算cam
        cam = self.cal_grad_cam(fusion_feature_map, fusion_grad_feature)
        # 绘制柱状图
        # 创建一个柱状集并添加数据
        set0 = QBarSet("Bar1")
        set0 << 1 << 2 << 3 << 4 << 5

        # 创建一个柱状序列并添加柱状集
        series = QBarSeries()
        series.append(set0)

        # 创建一个图表，添加序列
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Simple Bar Chart Example")

        # 创建一个数值轴，设置其最小值、最大值和步长
        axisX = QValueAxis()
        axisX.setRange(0, 5)
        axisX.setTickInterval(1)

        # 将数值轴添加到图表中
        chart.setAxisX(axisX, series)

        # 创建一个图表视图，设置抗锯齿渲染并设置图表
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.addWidget(chart_view)

        self.widget_2.setLayout(layout)


