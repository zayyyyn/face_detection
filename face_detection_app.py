import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from facenet_pytorch import MTCNN
import os

mtcnn = MTCNN(keep_all=True)
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_accuracy(gt_boxes, pred_boxes, iou_threshold=0.5):
    TP = 0
    matched_gt = set()
    matched_pred = set()
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            if compute_iou(gt, pred) >= iou_threshold:
                TP += 1
                matched_gt.add(i)
                matched_pred.add(j)
                break
    FP = len(pred_boxes) - len(matched_pred)
    FN = len(gt_boxes) - len(matched_gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, TP, FP, FN

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection Accuracy Comparison")
        self.root.geometry("1000x800")

        self.display_width = 800
        self.display_height = 500

        self.canvas = tk.Canvas(root, width=self.display_width, height=self.display_height, bg="gray")
        self.canvas.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_load_folder = tk.Button(btn_frame, text="Load Image Folder", command=self.load_folder)
        self.btn_load_folder.grid(row=0, column=0, padx=10)

        self.btn_next = tk.Button(btn_frame, text="▶ Next Image", command=self.next_image, state=tk.DISABLED)
        self.btn_next.grid(row=0, column=1, padx=10)

        self.btn_prev = tk.Button(btn_frame, text="◀ Previous Image", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.grid(row=0, column=2, padx=10)

        self.btn_clear = tk.Button(btn_frame, text="Clear Manual Boxes", command=self.clear_gt_boxes, state=tk.DISABLED)
        self.btn_clear.grid(row=0, column=3, padx=10)

        self.model_var = tk.StringVar()
        self.model_selector = ttk.Combobox(btn_frame, textvariable=self.model_var, state="readonly")
        self.model_selector['values'] = ("MTCNN", "MediaPipe", "Faster R-CNN")
        self.model_selector.current(0)
        self.model_selector.grid(row=0, column=4, padx=10)

        self.btn_detect = tk.Button(btn_frame, text="Detect & Compare", command=self.detect_and_compare, state=tk.DISABLED)
        self.btn_detect.grid(row=0, column=5, padx=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
        self.result_label.pack(pady=10)

        self.img_folder = None
        self.img_files = []
        self.index = 0
        self.image = None
        self.tk_image = None
        self.gt_boxes = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.img_folder = folder
            self.img_files = sorted([
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            if not self.img_files:
                messagebox.showerror("Error", "Folder does not contain images")
                return
            self.index = 0
            self.gt_boxes = []
            self.load_image()
            self.btn_next.config(state=tk.NORMAL if len(self.img_files) > 1 else tk.DISABLED)
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_clear.config(state=tk.NORMAL)
            self.btn_detect.config(state=tk.NORMAL)
            self.result_label.config(text="Draw ground truth face boxes manually on the image")

    def load_image(self):
        self.gt_boxes.clear()
        self.canvas.delete("all")
        img_path = os.path.join(self.img_folder, self.img_files[self.index])
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.image = img
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image
        self.result_label.config(text=f"Image: {self.img_files[self.index]} - Draw ground truth boxes (click and drag)")

    def start_draw(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2, tags="rect")

    def draw_rect(self, event):
        if self.drawing:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def end_draw(self, event):
        self.drawing = False
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        w, h = x2 - x1, y2 - y1
        if w > 10 and h > 10:
            self.gt_boxes.append([x1, y1, w, h])
        else:
            self.canvas.delete(self.rect)
        self.rect = None

    def clear_gt_boxes(self):
        self.gt_boxes.clear()
        self.canvas.delete("rect")
        self.result_label.config(text="Manual boxes cleared. Draw again.")

    def next_image(self):
        if self.index < len(self.img_files) - 1:
            self.index += 1
            self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def detect_and_compare(self):
        if not self.image or not self.gt_boxes:
            messagebox.showwarning("Warning", "Please draw the ground truth face boxes first.")
            return

        img_array = np.array(self.image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_tensor = F.to_tensor(self.image)
        w, h = self.image.size

        scale_x = img_array.shape[1] / w
        scale_y = img_array.shape[0] / h
        gt_scaled = [[int(x * scale_x), int(y * scale_y), int(wb * scale_x), int(hb * scale_y)] for (x, y, wb, hb) in self.gt_boxes]

        selected_model = self.model_var.get()
        pred_boxes = []

        if selected_model == "MTCNN":
            boxes_mtcnn, _ = mtcnn.detect(img_array)
            if boxes_mtcnn is not None:
                for box in boxes_mtcnn:
                    x1, y1, x2, y2 = map(int, box)
                    pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
        elif selected_model == "MediaPipe":
            results = mp_face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w_rel, h_rel = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                    x1 = int(x * w)
                    y1 = int(y * h)
                    pred_boxes.append([x1, y1, int(w_rel * w), int(h_rel * h)])
        elif selected_model == "Faster R-CNN":
            with torch.no_grad():
                outputs = faster_rcnn([img_tensor])[0]
            for i, score in enumerate(outputs['scores']):
                if score > 0.5:
                    x1, y1, x2, y2 = map(int, outputs['boxes'][i])
                    pred_boxes.append([x1, y1, x2 - x1, y2 - y1])

        p, r, f1, _, _, _ = evaluate_accuracy(gt_scaled, pred_boxes)
        self.result_label.config(text=f"Model: {selected_model}\nF1 = {f1:.2f}, Precision = {p:.2f}, Recall = {r:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
