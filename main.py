# YOLO-Lite 🚀
import os
os.environ['YOLO_SETTINGS_DISABLED'] = 'True'
from yololite import YOLOLite
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，必须在import pyplot之前
# 加载预训练模型
model = YOLOLite("yolo26l.pt")  # 加载预训练模型权重
  # 如果有这个属性会显示yaml路径

# 不使用预训练模型，会导致损失难以下降
# model = YOLOLite("yololite3d/cfg/yolo11.yaml")
if __name__ == '__main__':
    # 如果是打包成 exe，需要这行
    # from multiprocessing import freeze_support
    # freeze_support()
   # results = model.train(data="voc.yaml", epochs=1, imgsz=640)
# 训练coco8


#无标签训练
  """


     results = model.train(
            data="data_labeled.yaml",               # 有标签数据集配置文件路径
            unlabeldata="data_unlabel.yaml", # 无标签数据集配置文件路径
            epochs=150,                      # 训练周期数
            imgsz=640,                     # 输入图像尺寸
            batch=4,                      # 批次大小
            device="cuda"
        )


    #全监督训练
  results = model.train(
          data="data.yaml",               # 有标签数据集配置文件路径
          #unlabeldata="data_unlabel.yaml", # 无标签数据集配置文件路径
          epochs=175,                      # 训练周期数
          imgsz=640,                     # 输入图像尺寸
          batch=8,                      # 批次大小
          device="cuda"
      )

  """

# 推理
# results = model(["boats.jpg"])
# print(results[0].boxes)

  results = model.train(
          data="data_labeled.yaml",               # 有标签数据集配置文件路径
          unlabeldata="data_unlabel.yaml", # 无标签数据集配置文件路径
          epochs=175,                      # 训练周期数
          imgsz=640,                     # 输入图像尺寸
          batch=4,                      # 批次大小
          device="cuda"
      )