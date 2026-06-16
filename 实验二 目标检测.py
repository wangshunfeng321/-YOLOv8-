import cv2
import numpy as np
import time
import os
from ultralytics import YOLO


class ObjectDetector:
    """基础目标检测器（使用YOLOv8）"""

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        初始化目标检测器
        参数:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("模型加载成功")

    def detect_image(self, image_path):
        """
        检测图像中的目标
        参数:
            image_path: 图像路径
        返回:
            results: 检测结果
        """
        results = self.model(image_path, conf=self.conf_threshold)
        return results

    def visualize_results(self, image_path, results):
        """
        可视化检测结果
        参数:
            image_path: 原始图像路径
            results: 检测结果
        """
        image = cv2.imread(image_path)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # 提取框的坐标 map 函数将浮点数转换为整数类型
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{result.names[cls]} {conf:.2f}'

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


class RealtimeDetector:
    """实时视频流目标检测器"""

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        初始化实时检测器
        参数:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        print(f"正在加载实时检测模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("实时检测模型加载成功")


    def detect_video(self, video_source=0):
        """
        实时检测视频流
        参数:
            video_source: 视频源(0为摄像头,或视频文件路径)
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {video_source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height} @ {fps}fps")
        print("按 'q' 键退出检测")

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或读取失败")
                break

            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            curr_time = time.time()
            fps_display = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(annotated_frame, f'FPS: {fps_display:.1f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Realtime Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class OptimizedDetector:
    """优化的目标检测器（使用OpenCV DNN + ONNX）"""

    def __init__(self, model_path='yolov8n.onnx', conf_threshold=0.5):
        """
        初始化优化检测器
        参数:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
        """
        if not os.path.exists(model_path):
            print(f"ONNX模型不存在，正在从 YOLOv8 模型导出...")
            self._export_onnx() # 从YOLOv8模型导出ONNX格式

        print(f"正在加载优化模型: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.classes = self._load_classes()
        print("优化模型加载成功")

    def _export_onnx(self):
        """从YOLOv8模型导出ONNX格式"""
        model = YOLO('yolov8n.pt')
        model.export(format='onnx')
        print("ONNX模型导出成功")

    def _load_classes(self):
        """加载COCO数据集类别名称"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return coco_classes

    def preprocess(self, image, input_size=(640, 640)):
        """图像预处理"""
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            input_size,
            swapRB=True,
            crop=False
        )
        return blob

    def postprocess(self, outputs, image_shape):
        """后处理检测结果"""
        boxes = []
        confidences = []
        class_ids = []

        # YOLOv8输出格式: [batch, 84, 8400] -> 转置为 [batch, 8400, 84]
        output = outputs[0]
        output = output[0].transpose()  # [8400, 84]

        # 解析每个检测
        for detection in output:
            # detection格式: [x, y, w, h, class0_conf, class1_conf, ..., class79_conf]
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_threshold:
                # YOLOv8输出是中心点格式
                center_x = int(detection[0])
                center_y = int(detection[1])
                width = int(detection[2])
                height = int(detection[3])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)

        final_boxes = []
        final_confidences = []
        final_class_ids = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids

    def detect(self, image):
        """检测图像中的目标"""
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes, confidences, class_ids = self.postprocess(outputs, image.shape)
        return boxes, confidences, class_ids

    def visualize(self, image, boxes, confidences, class_ids):
        """可视化检测结果"""
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = f'{self.classes[class_ids[i]]}: {confidences[i]:.2f}'

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def detect_video(self, video_source=0):
        """使用优化模型进行实时检测"""
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {video_source}")
            return

        print("按 'q' 键退出检测")

        prev_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或读取失败")
                break

            boxes, confidences, class_ids = self.detect(frame)
            result_frame = self.visualize(frame.copy(), boxes, confidences, class_ids)

            frame_count += 1
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0

            cv2.putText(result_frame, f'FPS: {fps:.1f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Optimized Detection', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数：提供交互式菜单"""

    while True:
        print("\n请选择功能:")
        print("1. 图像目标检测（YOLOv8）")
        print("2. 实时视频检测（YOLOv8）")
        print("3. 优化模型实时检测（OpenCV DNN + ONNX）")
        print("4. 退出")

        choice = input("\n请输入选项 (1-4): ").strip()

        if choice == '1':
            image_path = input("请输入图像路径 (直接回车使用默认测试图像): ").strip()
            if not image_path:
                print("未提供图像路径，请准备测试图像")
                continue

            if not os.path.exists(image_path):
                print(f"错误: 图像文件不存在: {image_path}")
                continue

            detector = ObjectDetector(conf_threshold=0.5)
            results = detector.detect_image(image_path)
            result_image = detector.visualize_results(image_path, results)

            cv2.imshow('Detection Result', result_image)
            print("按任意键关闭图像窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif choice == '2':
            source = input("请输入视频源 (0=摄像头, 或视频文件路径, 直接回车使用摄像头): ").strip()
            video_source = 0 if not source else (int(source) if source.isdigit() else source)

            detector = RealtimeDetector(conf_threshold=0.5)
            detector.detect_video(video_source)

        elif choice == '3':
            source = input("请输入视频源 (0=摄像头, 或视频文件路径, 直接回车使用摄像头): ").strip()
            video_source = 0 if not source else (int(source) if source.isdigit() else source)

            detector = OptimizedDetector(conf_threshold=0.5)
            detector.detect_video(video_source)

        elif choice == '4':
            print("退出系统")
            break

        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()