from osgeo import gdal
from osgeo import ogr
import os
os.environ['PROJ_LIB'] = r'D:\Anaconda\envs\dl_py310\Lib\site-packages\osgeo\data\proj'

train_samples_file = r"D:\Work_Space\Software_Forest_Disturbance\data_examples\data_import\train_samples.shp"
test_samples_file = r"D:\Work_Space\Software_Forest_Disturbance\data_examples\data_import\test_samples.shp"
Sentinel2_reference_file = r"D:\Work_Space\Software_Forest_Disturbance\data_examples\data_import\B03.tif"
output_path = r"D:\Work_Space\Software_Forest_Disturbance\temp_files"

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
    raster = gdal.Open(Sentinel2_reference_file)
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

if __name__ == "__main__":
    shp2raster(train_samples_file, Sentinel2_reference_file, output_path)
    shp2raster(test_samples_file, Sentinel2_reference_file, output_path)