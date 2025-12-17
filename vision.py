import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os
import shutil
import threading
import yaml
import time
import datetime
import psutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mpl_style
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
mpl_style.use('dark_background')
folder = "hanash_data"
sub_folders = [f"{folder}/train/images", f"{folder}/train/labels", 
               f"{folder}/val/images", f"{folder}/val/labels"]
for p in sub_folders:
    os.makedirs(p, exist_ok=True)
class TrainingCallback:
    def __init__(self, app_ref, total_epochs):
        self.app = app_ref
        self.epoch_count = 0
        self.total_epochs = total_epochs
        self.start_time = time.time()
    def on_train_epoch_end(self, trainer):
        self.epoch_count += 1
        metrics = trainer.metrics
        loss = trainer.loss_items[0].item() if hasattr(trainer, 'loss_items') else 0.0
        map50 = metrics.get("metrics/mAP50(B)", 0)
        
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.epoch_count
        remaining = (self.total_epochs - self.epoch_count) * avg_time
        time_str = str(datetime.timedelta(seconds=int(remaining)))
        self.app.after(1, lambda: self.app.update_training_ui(
            self.epoch_count, loss, map50, time_str
        ))
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HANASH VISION DASHBOARD ")
        self.geometry("1380x850")
        self.active = True
        self.app_mode = "IDLE" 
        self.cam_source = None
        self.items = [] 
        self.counter = 0
        self.model_path = "yolov8n.pt" 
        self.recording = False
        self.target_epochs = 25
        self.train_losses = []
        self.train_maps = []
        self.train_epochs = []
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#0f0f0f")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.logo = ctk.CTkLabel(self.sidebar, text="HANASH\nNEURAL HUB", font=ctk.CTkFont(size=26, weight="bold"))
        self.logo.pack(pady=(30, 20))
        self.input_frame = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a")
        self.input_frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(self.input_frame, text="OBJECT CLASS NAME", font=("Arial", 10, "bold"), text_color="#777").pack(anchor="w", padx=10, pady=(10,0))
        self.entry_name = ctk.CTkEntry(self.input_frame, placeholder_text="e.g. My Watch")
        self.entry_name.pack(fill="x", padx=10, pady=(5, 15))
        self.btn_capture = ctk.CTkButton(self.sidebar, text="📷 START CAPTURE", height=45, fg_color="#D35400", hover_color="#E67E22", font=ctk.CTkFont(weight="bold"), command=self.do_capture)
        self.btn_capture.pack(padx=15, pady=5, fill="x")
        self.btn_train = ctk.CTkButton(self.sidebar, text="🧠 TRAIN MODEL (5m)", height=45, fg_color="#2980B9", hover_color="#3498DB", font=ctk.CTkFont(weight="bold"), command=self.do_train)
        self.btn_train.pack(padx=15, pady=5, fill="x")
        self.btn_detect = ctk.CTkButton(self.sidebar, text="🟢 LIVE INFERENCE", height=45, fg_color="#27AE60", hover_color="#2ECC71", font=ctk.CTkFont(weight="bold"), command=self.do_detect)
        self.btn_detect.pack(padx=15, pady=(20, 10), fill="x")
        self.btn_clear = ctk.CTkButton(self.sidebar, text="🗑️ DELETE ALL DATA", height=30, fg_color="#333", hover_color="#C0392B", font=ctk.CTkFont(size=11), command=self.clear_dataset)
        self.btn_clear.pack(padx=15, pady=5, fill="x")
        self.console = ctk.CTkTextbox(self.sidebar, height=120, fg_color="#000", text_color="#0f0", font=("Consolas", 11))
        self.console.pack(padx=15, pady=15, fill="both", side="bottom")
        self.log_msg("System Ready. Waiting for input...")
        self.tab_view = ctk.CTkTabview(self, fg_color="transparent")
        self.tab_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)
        self.tab_cam = self.tab_view.add("👁️ VISION FEED")
        self.tab_train = self.tab_view.add("📊 TRAINING HUD")
        self.tab_cam.grid_columnconfigure(0, weight=1)
        self.tab_cam.grid_rowconfigure(0, weight=1)
        self.cam_frame = ctk.CTkFrame(self.tab_cam, fg_color="#000")
        self.cam_frame.grid(row=0, column=0, sticky="nsew")
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="")
        self.cam_label.pack(fill="both", expand=True)
        self.setup_hud()
        self.init_cam()
        threading.Thread(target=self.system_monitor_loop, daemon=True).start()
    def setup_hud(self):
        self.tab_train.grid_columnconfigure(0, weight=1)
        self.tab_train.grid_columnconfigure(1, weight=1)
        self.stat_frame = ctk.CTkFrame(self.tab_train, fg_color="transparent")
        self.stat_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.card_epoch = self.create_card(self.stat_frame, "EPOCH", "0/0")
        self.card_loss = self.create_card(self.stat_frame, "LOSS", "0.00")
        self.card_map = self.create_card(self.stat_frame, "PRECISION", "0%")
        self.card_eta = self.create_card(self.stat_frame, "ETA", "00:00")
        self.graph_frame = ctk.CTkFrame(self.tab_train, fg_color="#181818")
        self.graph_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=10)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)
        self.fig.patch.set_facecolor('#181818')
        for ax, title in [(self.ax1, "Training Loss"), (self.ax2, "Accuracy (mAP)")]:
            ax.set_title(title, color='white', fontsize=9)
            ax.set_facecolor('#222')
            ax.tick_params(colors='white', labelsize=7) 
        self.line_loss, = self.ax1.plot([], [], color='#e74c3c')
        self.line_map, = self.ax2.plot([], [], color='#2ecc71')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.bot_frame = ctk.CTkFrame(self.tab_train, fg_color="transparent")
        self.bot_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        self.lbl_cpu = ctk.CTkLabel(self.bot_frame, text="CPU LOAD", font=("Arial", 10, "bold"))
        self.lbl_cpu.pack(anchor="w")
        self.bar_cpu = ctk.CTkProgressBar(self.bot_frame, height=8, progress_color="#3498db")
        self.bar_cpu.pack(fill="x", pady=(0, 5))
        self.lbl_ram = ctk.CTkLabel(self.bot_frame, text="RAM USAGE", font=("Arial", 10, "bold"))
        self.lbl_ram.pack(anchor="w")
        self.bar_ram = ctk.CTkProgressBar(self.bot_frame, height=8, progress_color="#9b59b6")
        self.bar_ram.pack(fill="x", pady=(0, 15))
        self.lbl_prog = ctk.CTkLabel(self.bot_frame, text="TOTAL TRAINING PROGRESS", font=("Arial", 12, "bold"))
        self.lbl_prog.pack(anchor="w")
        self.main_prog = ctk.CTkProgressBar(self.bot_frame, height=25, progress_color="#27ae60")
        self.main_prog.pack(fill="x")
        self.main_prog.set(0)
    def create_card(self, parent, title, val):
        f = ctk.CTkFrame(parent, fg_color="#252525")
        f.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkLabel(f, text=title, font=("Arial", 10, "bold"), text_color="#888").pack(pady=(5,0))
        lbl = ctk.CTkLabel(f, text=val, font=("Arial", 22, "bold"))
        lbl.pack(pady=(0,5))
        f.val_lbl = lbl
        return f
    def log_msg(self, txt):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.console.insert("end", f"[{ts}] {txt}\n")
        self.console.see("end")
    def system_monitor_loop(self):
        while self.active:
            cpu = psutil.cpu_percent() / 100
            ram = psutil.virtual_memory().percent / 100
            try:
                self.bar_cpu.set(cpu)
                self.bar_ram.set(ram)
            except: pass
            time.sleep(1)

    def init_cam(self):
        self.cam_source = cv2.VideoCapture(0)
        self.cam_source.set(3, 1280)
        self.cam_source.set(4, 720)
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        ai = YOLO(self.model_path)
        while self.active:
            ret, frame = self.cam_source.read()
            if not ret: continue

            if self.app_mode == "CAPTURE":
                h, w, _ = frame.shape
                d = 320
                sx, sy = (w-d)//2, (h-d)//2
                ex, ey = sx+d, sy+d
                
                color = (0, 255, 255) if self.recording else (100, 100, 100)
                cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
                if self.recording:
                    cv2.putText(frame, f"REC: {self.counter}", (sx, sy-10), 0, 0.7, color, 2)
                    self.save_frame(frame, sx, sy, ex, ey)

            elif self.app_mode == "DETECT":
                res = ai(frame, verbose=False, conf=0.55)
                frame = res[0].plot()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            
            try:
                w_app = self.cam_label.winfo_width()
                h_app = self.cam_label.winfo_height()
                if w_app > 10:
                    pil = pil.resize((w_app, h_app), Image.Resampling.NEAREST)
                tk_img = ImageTk.PhotoImage(image=pil)
                self.cam_label.configure(image=tk_img)
                self.cam_label.image = tk_img
            except: pass
            time.sleep(0.03)

    def do_capture(self):
        obj = self.entry_name.get()
        if not obj: return self.log_msg("⚠️ Name Required!")
        if obj not in self.items: self.items.append(obj)
        
        self.app_mode = "CAPTURE"
        self.recording = True
        self.counter = 0
        self.tab_view.set("👁️ VISION FEED")
        self.after(4000, self.end_capture)

    def end_capture(self):
        self.recording = False
        self.log_msg(f"✅ Captured {self.counter} images.")
        self.app_mode = "IDLE"

    def save_frame(self, img, x1, y1, x2, y2):
        name = self.entry_name.get()
        idx = self.items.index(name)
        fname = f"{name}_{time.time()}"
        cv2.imwrite(f"{folder}/train/images/{fname}.jpg", img)
        h, w, _ = img.shape
        cx, cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
        bw, bh = (x2-x1)/w, (y2-y1)/h
        with open(f"{folder}/train/labels/{fname}.txt", 'w') as f:
            f.write(f"{idx} {cx} {cy} {bw} {bh}")
        self.counter += 1

    def clear_dataset(self):
        self.log_msg("DELETING ALL DATASETS...")
        try:
            shutil.rmtree(folder)
            for p in sub_folders: os.makedirs(p, exist_ok=True)
            self.items = []
            self.train_losses, self.train_maps, self.train_epochs = [], [], []
            self.ax1.clear(); self.ax2.clear(); self.canvas.draw()
            self.main_prog.set(0)
            self.log_msg("System Reset Clean.")
        except Exception as e:
            self.log_msg(f"Error: {e}")

    def do_train(self):
        if not self.items: return self.log_msg("❌ No data to train!")
        self.tab_view.set("TRAINING HUD")
        self.log_msg("Initializing Neural Engine...")
        self.train_losses, self.train_maps, self.train_epochs = [], [], []
        self.ax1.cla(); self.ax2.cla(); self.canvas.draw()
        
        d = {'path': os.path.abspath(folder), 'train': 'train/images', 
             'val': 'train/images', 'nc': len(self.items), 'names': self.items}
        with open(f"{folder}/data.yaml", 'w') as f: yaml.dump(d, f)
        threading.Thread(target=self.process_training).start()

    def process_training(self):
        try:
            self.app_mode = "TRAINING"
            m = YOLO('yolov8n.pt')
            cb = TrainingCallback(self, self.target_epochs)
            m.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
            m.train(data=f"{folder}/data.yaml", epochs=self.target_epochs, 
                    imgsz=320, batch=4, project="hanash_runs", name="pro_model", exist_ok=True)
            self.model_path = "hanash_runs/pro_model/weights/best.pt"
            self.log_msg("🚀 Training Complete! Model Live.")
            self.app_mode = "IDLE"
        except Exception as e:
            self.log_msg(f"Error: {e}")
            self.app_mode = "IDLE"

    def update_training_ui(self, epoch, loss, map50, eta):
        self.card_epoch.val_lbl.configure(text=f"{epoch}/{self.target_epochs}")
        self.card_loss.val_lbl.configure(text=f"{loss:.3f}")
        self.card_map.val_lbl.configure(text=f"{map50:.1%}")
        self.card_eta.val_lbl.configure(text=eta)
        self.train_epochs.append(epoch)
        self.train_losses.append(loss)
        self.train_maps.append(map50)
        self.ax1.clear(); self.ax1.plot(self.train_epochs, self.train_losses, color='#e74c3c')
        self.ax1.set_title("Training Loss", color='white', fontsize=9); self.ax1.set_facecolor('#222')
        self.ax2.clear(); self.ax2.plot(self.train_epochs, self.train_maps, color='#2ecc71')
        self.ax2.set_title("Accuracy (mAP)", color='white', fontsize=9); self.ax2.set_facecolor('#222')
        self.canvas.draw()
        self.main_prog.set(epoch / self.target_epochs)

    def do_detect(self):
        self.tab_view.set("VISION FEED")
        self.app_mode = "DETECT"
        self.log_msg("Live Inference Started.")

    def close_app(self):
        self.active = False
        if self.cam_source: self.cam_source.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()