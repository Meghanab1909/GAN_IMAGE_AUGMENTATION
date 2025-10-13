import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch.nn as nn

class SyntheticVisionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Vision: GANs Evaluation")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # Classifier
        self.classifier = None
        self.classifier_loaded = False

        # Dataset
        self.dataset_folder = r"C:\Users\HP\Downloads\FlowerGAN_Project\flowers"
        self.dataset_images = []
        self.current_dataset_idx = 0
        self.total_dataset_images = 0

        # GAN images folder
        self.gan_folder = r"C:\Users\HP\Downloads\FlowerGAN_Project\gan_samples"
        self.generated_images = []

        self.flower_types = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

        self.build_ui()
        self.load_dataset_on_startup()
        self.load_classifier()

    # ---------------- Build UI ----------------
    def build_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel: Dataset
        control_frame = tk.Frame(main_frame, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,10), pady=0)

        tk.Label(control_frame, text="Original Dataset Preview", font=("Arial",12,"bold"),
                 bg="#ffffff", fg="#34495e").pack(pady=(10,10))
        self.dataset_canvas = tk.Canvas(control_frame, width=250, height=250, bg="#ecf0f1",
                                        highlightthickness=1, highlightbackground="#bdc3c7")
        self.dataset_canvas.pack(pady=10, padx=20)

        nav_frame = tk.Frame(control_frame, bg="#ffffff")
        nav_frame.pack(pady=10)
        self.prev_btn = tk.Button(nav_frame, text="← Previous", command=self.prev_dataset_image,
                                  bg="#95a5a6", fg="white", font=("Arial",9,"bold"),
                                  padx=10, pady=5, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.dataset_label = tk.Label(nav_frame, text="0/0", font=("Arial",10), bg="#ffffff")
        self.dataset_label.pack(side=tk.LEFT, padx=10)
        self.next_btn = tk.Button(nav_frame, text="Next →", command=self.next_dataset_image,
                                  bg="#95a5a6", fg="white", font=("Arial",9,"bold"),
                                  padx=10, pady=5, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.generate_btn = tk.Button(control_frame, text="Display GAN Images",
                                      command=self.generate_images,
                                      bg="#3498db", fg="white", font=("Arial",12,"bold"),
                                      relief=tk.RAISED, borderwidth=3, cursor="hand2", padx=20, pady=10)
        self.generate_btn.pack(pady=20, padx=20, fill=tk.X)

        # Right panel: GAN images + metrics
        display_frame = tk.Frame(main_frame, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0)

        tk.Label(display_frame, text="Generated GAN Images", font=("Arial",14,"bold"),
                 bg="#ffffff", fg="#34495e").pack(pady=(20,10))
        self.status_label = tk.Label(display_frame, text="Click 'Display GAN Images' to begin",
                                     font=("Arial",10,"italic"), bg="#ffffff", fg="#7f8c8d")
        self.status_label.pack(pady=10)
        self.images_frame = tk.Frame(display_frame, bg="#ffffff")
        self.images_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Metrics
        tk.Label(control_frame, text="Evaluation Metrics", font=("Arial",14,"bold"),
                 bg="#ffffff", fg="#34495e").pack(pady=(10,5))
        self.metrics_text = tk.Text(control_frame, width=40, height=8, font=("Arial",10), bg="#ecf0f1")
        self.metrics_text.pack(pady=10, padx=20)
        self.cm_canvas_frame = tk.Frame(display_frame, bg="#ffffff")
        self.cm_canvas_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    # ---------------- Dataset ----------------
    def load_dataset_on_startup(self):
        exts = ('.png','.jpg','.jpeg','.bmp','.gif')
        self.dataset_images = [os.path.join(r,f)
                               for r,d,files in os.walk(self.dataset_folder)
                               for f in files if f.lower().endswith(exts)]
        self.dataset_images.sort()
        if self.dataset_images:
            self.total_dataset_images = len(self.dataset_images)
            self.current_dataset_idx = 0
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            self.update_dataset_preview()

    def update_dataset_preview(self):
        self.dataset_canvas.delete("all")
        if self.dataset_images:
            img = Image.open(self.dataset_images[self.current_dataset_idx])
            img.thumbnail((240,240))
            photo = ImageTk.PhotoImage(img)
            self.dataset_canvas.create_image(125,125,image=photo, anchor=tk.CENTER)
            self.dataset_canvas.image = photo
            self.dataset_label.config(text=f"{self.current_dataset_idx+1}/{self.total_dataset_images}")

    def prev_dataset_image(self):
        if self.dataset_images:
            self.current_dataset_idx = (self.current_dataset_idx-1)%self.total_dataset_images
            self.update_dataset_preview()

    def next_dataset_image(self):
        if self.dataset_images:
            self.current_dataset_idx = (self.current_dataset_idx+1)%self.total_dataset_images
            self.update_dataset_preview()

    # ---------------- Load classifier ----------------
    def load_classifier(self):
        try:
            num_classes = len(self.flower_types)
            self.classifier = models.efficientnet_b0(weights=None)
            self.classifier.classifier[1] = nn.Linear(self.classifier.classifier[1].in_features, num_classes)

            checkpoint_path = r"C:\Users\HP\Downloads\FlowerGAN_Project\effnet_b0_best.pth"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # If checkpoint saved as a dict
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            self.classifier.load_state_dict(checkpoint)
            self.classifier.to(self.device).eval()
            self.classifier_loaded = True
            self.status_label.config(text="Classifier loaded! Ready to evaluate images.", fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error loading classifier", str(e))
            self.classifier_loaded = False

    # ---------------- Display GAN images and evaluate ----------------
    def generate_images(self):
        if not self.classifier_loaded:
            messagebox.showwarning("Warning", "Classifier not loaded!")
            return

        exts = ('.png','.jpg','.jpeg','.bmp','.gif')
        self.generated_images = [os.path.join(self.gan_folder,f)
                                for f in os.listdir(self.gan_folder)
                                if f.lower().endswith(exts)]
        self.generated_images.sort()
        
        if not self.generated_images:
            messagebox.showwarning("Warning", "No GAN images found in folder!")
            return

        self.display_generated_images()
        self.evaluate_gan_images()

    def display_generated_images(self):
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        num_images = len(self.generated_images)
        cols = min(5,num_images)
        rows = (num_images + cols -1)//cols
        for i,img_path in enumerate(self.generated_images):
            row = i//cols
            col = i%cols
            frame = tk.Frame(self.images_frame, bg="#ecf0f1", relief=tk.RAISED, borderwidth=2)
            frame.grid(row=row,column=col,padx=10,pady=10,sticky="nsew")
            try:
                img = Image.open(img_path)
                img.thumbnail((120,120))
                photo = ImageTk.PhotoImage(img)
                canvas = tk.Canvas(frame,width=120,height=120,bg="#34495e",highlightthickness=0)
                canvas.pack(padx=5,pady=5)
                canvas.create_image(60,60,image=photo)
                canvas.image = photo
            except:
                canvas = tk.Canvas(frame,width=120,height=120,bg="#e74c3c",highlightthickness=0)
                canvas.pack(padx=5,pady=5)
                canvas.create_text(100,100,text="Error\nLoading Image", fill="white",
                                   font=("Arial",12,"bold"), justify=tk.CENTER)
            tk.Label(frame,text=f"Augmented Image {i+1}",font=("Arial",9),bg="#ecf0f1").pack(pady=(0,5))
        self.status_label.config(text=f"Displaying {num_images} GAN images", fg="#27ae60")

    # ---------------- Evaluate GAN images ----------------
    def evaluate_gan_images(self):
        if not self.generated_images:
            return

        class_counts = {flower: 0 for flower in self.flower_types}

        for img_path in self.generated_images:
            img = Image.open(img_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classifier(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                class_counts[self.flower_types[pred]] += 1

        # Display class counts
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "Predicted Class Distribution:\n")
        for flower, count in class_counts.items():
            self.metrics_text.insert(tk.END, f"{flower}: {count}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticVisionUI(root)
    root.mainloop()