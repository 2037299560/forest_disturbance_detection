from osgeo import gdal
import os
os.environ['PROJ_LIB'] = r'D:\Anaconda\envs\dl_py310\Lib\site-packages\osgeo\data\proj'

# 窗口的行列号范围
start_row, start_col = 2294, 326
end_row, end_col = 2998, 1187

for band in [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "VV", "VH"
]:
    # 打开原始影像
    file_name = r"E:\time_series_deep_learning_forest_distrub\src\普洱\data_preprocess\tmp_data_store\47QQF\2022\2022_{}.tif".format(band)
    print(file_name)
    ds = gdal.Open(file_name)

    # 读取窗口的数据
    window = ds.ReadAsArray(start_col, start_row, end_col-start_col, end_row-start_row)

    # 创建新的数据集
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(f"./data_examples/data_import/{band}.tif", end_col-start_col, end_row-start_row, ds.RasterCount, ds.GetRasterBand(1).DataType)

    # 设置新的数据集的地理变换和投影信息
    geotrans = list(ds.GetGeoTransform())
    geotrans[0] = geotrans[0] + start_col * geotrans[1]
    geotrans[3] = geotrans[3] + start_row * geotrans[5]
    out_ds.SetGeoTransform(geotrans)
    out_ds.SetProjection(ds.GetProjection())

    # 将窗口的数据写入新的数据集
    for i in range(ds.RasterCount):
        out_band = out_ds.GetRasterBand(i+1)
        out_band.WriteArray(window[i])

    # 保存新的数据集
    out_ds.FlushCache()
    out_ds = None