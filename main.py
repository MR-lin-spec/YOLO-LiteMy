# YOLO-Lite 🚀

from yololite import YOLOLite

# 加载预训练模型
model = YOLOLite("yolo11n.pt")

# 不使用预训练模型，会导致损失难以下降
# model = YOLOLite("yololite3d/cfg/yolo11.yaml")
if __name__ == '__main__':
    # 如果是打包成 exe，需要这行
    # from multiprocessing import freeze_support
    # freeze_support()
    
    results = model.train(data="voc.yaml", epochs=1, imgsz=640)
# 训练coco8


# 推理
# results = model(["boats.jpg"])
# print(results[0].boxes)
