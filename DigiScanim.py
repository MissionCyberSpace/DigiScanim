import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import math
from numba import njit

# ==========================================
# NUMBA JIT KERNEL (COMPILED C-LEVEL MATH)
# ==========================================
@njit(fastmath=True)
def calculate_raster_numba(
    t, raster_y, target_w, target_h, src_w, src_h,
    global_time, morph, scanlines, intensity, persistence, samples,
    rx, ry, rz, fx, fy, fz,
    theta, trans_x, trans_y, scale_x, scale_y,
    sz_x, gap_x, cx, sx, spd_x,
    sz_y, gap_y, cy, sy, spd_y,
    buffer_state, colors
):
    # Scale beam intensity based on current sample density
    color_mult = intensity / (samples / 15000.0)
    
    # Pre-compute rotation matrix sines and cosines
    cos_rx, sin_rx = math.cos(rx), math.sin(rx)
    cos_ry, sin_ry = math.cos(ry), math.sin(ry)
    cos_rz, sin_rz = math.cos(rz), math.sin(rz)
    
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Fade old frame for CRT persistence trail
    buffer_state *= persistence

    for i in range(len(t)):
        raster_x_val = (t[i] * scanlines) % 1.0
        
        u = raster_x_val * 2.0 - 1.0
        v = raster_y[i] * 2.0 - 1.0
        
        # 2D Rotation
        u_rot = u * cos_theta - v * sin_theta
        v_rot = u * sin_theta + v * cos_theta
        
        # Isolated Pinch X (Modulo gap logic)
        dist_x = (v_rot + global_time * spd_x) % gap_x
        dist_x = dist_x - (gap_x / 2.0)
        norm_x = dist_x / sz_x
        wave_x = math.sin(norm_x * math.pi) if abs(norm_x) < 1.0 else 0.0
        mod_x = wave_x * sx if wave_x > 0 else wave_x * cx
        u_pinch = u_rot * (1.0 + mod_x)
        
        # Isolated Pinch Y (Modulo gap logic)
        dist_y = (u_rot + global_time * spd_y) % gap_y
        dist_y = dist_y - (gap_y / 2.0)
        norm_y = dist_y / sz_y
        wave_y = math.sin(norm_y * math.pi) if abs(norm_y) < 1.0 else 0.0
        mod_y = wave_y * sy if wave_y > 0 else wave_y * cy
        v_pinch = v_rot * (1.0 + mod_y)
        
        # Scale and Translate
        u_final = (u_pinch / scale_x) + trans_x
        v_final = (v_pinch / scale_y) + trans_y
        
        # Map back to source image
        src_x = int(min(max((u_final + 1.0) * 0.5 * (src_w - 1), 0), src_w - 1))
        src_y = int(min(max((v_final + 1.0) * 0.5 * (src_h - 1), 0), src_h - 1))

        # 3D Lissajous
        liss_x = math.sin(t[i] * fx * 2 * math.pi)
        liss_y = math.cos(t[i] * fy * 2 * math.pi)
        liss_z = math.sin(t[i] * fz * 2 * math.pi + global_time)

        # 3D Matrix Rotation (Unrolled for speed)
        x1 = liss_x
        y1 = liss_y * cos_rx - liss_z * sin_rx
        z1 = liss_y * sin_rx + liss_z * cos_rx

        x2 = x1 * cos_ry + z1 * sin_ry
        y2 = y1
        
        proj_x = x2 * cos_rz - y2 * sin_rz
        proj_y = x2 * sin_rz + y2 * cos_rz

        # Morph 3D back to 2D Raster and map to screen
        final_x = proj_x * (1.0 - morph) + (raster_x_val * 2.0 - 1.0) * morph
        final_y = proj_y * (1.0 - morph) + (raster_y[i] * 2.0 - 1.0) * morph

        screen_x = int(min(max((final_x + 1.0) * 0.5 * (target_w - 1), 0), target_w - 1))
        screen_y = int(min(max((final_y + 1.0) * 0.5 * (target_h - 1), 0), target_h - 1))

        # Additive blending
        c0 = colors[src_y, src_x, 0] * color_mult
        c1 = colors[src_y, src_x, 1] * color_mult
        c2 = colors[src_y, src_x, 2] * color_mult

        buffer_state[screen_y, screen_x, 0] += c0
        buffer_state[screen_y, screen_x, 1] += c1
        buffer_state[screen_y, screen_x, 2] += c2

# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
class ProductionScanimate:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.live_w, self.live_h = 800, 600
        self.export_w, self.export_h = 640, 480
        
        # Proxy system parameters
        self.live_samples = 40000  
        self.export_samples = 400000  
        self.t_live = np.linspace(0, 1, self.live_samples)
        self.t_export = np.linspace(0, 1, self.export_samples)
        
        self.t = self.t_live
        self.samples = self.live_samples
        self.raster_y = self.t
        
        self.img = cv2.imread("logo.png")
        if self.img is not None:
            self.src_h, self.src_w = self.img.shape[:2]
        else:
            self.img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.img, "NO PNG", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            self.src_h, self.src_w = 480, 640

        self.use_camera = False
        self.global_time = 0.0
        self.spin_x_acc = 0.0
        self.spin_y_acc = 0.0
        self.spin_z_acc = 0.0

        self.phosphor_buffer = np.zeros((self.live_h, self.live_w, 3), dtype=np.float32)

        self.keyframes = []
        self.playing = False
        self.play_start_time = 0
        self.transition_duration = 3.0
        self.last_time = time.time()

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Production Scanimate - Numba Engine")
        self.root.geometry("1400x950") 

        # Layout setup
        self.left_panel = tk.Frame(self.root, width=540)
        self.left_panel.pack(side="left", fill="y")
        self.left_panel.pack_propagate(False)

        self.canvas = tk.Canvas(self.left_panel)
        self.scrollbar = tk.Scrollbar(self.left_panel, orient="vertical", command=self.canvas.yview)
        self.ui_frame = tk.Frame(self.canvas)

        self.ui_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.ui_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.vid_label = tk.Label(self.root, bg="black")
        self.vid_label.pack(side="right", fill="both", expand=True)

        # Variables
        self.vars = {
            "morph": tk.DoubleVar(value=0.0),
            "scanlines": tk.IntVar(value=150),
            
            "freq_x": tk.DoubleVar(value=3.0),
            "freq_y": tk.DoubleVar(value=2.0),
            "freq_z": tk.DoubleVar(value=2.0),
            
            "spin_x": tk.DoubleVar(value=0.0),
            "spin_y": tk.DoubleVar(value=0.5),
            "spin_z": tk.DoubleVar(value=0.0),
            
            "trans_x": tk.DoubleVar(value=0.0),
            "trans_y": tk.DoubleVar(value=0.0),
            "scale_x": tk.DoubleVar(value=1.0),
            "scale_y": tk.DoubleVar(value=1.0),
            "rot_2d": tk.DoubleVar(value=0.0),
            
            "crush_x": tk.DoubleVar(value=0.0),
            "stretch_x": tk.DoubleVar(value=0.0),
            "pinch_sz_x": tk.DoubleVar(value=0.5),
            "pinch_gap_x": tk.DoubleVar(value=2.0),
            "pinch_spd_x": tk.DoubleVar(value=1.0),
            
            "crush_y": tk.DoubleVar(value=0.0),
            "stretch_y": tk.DoubleVar(value=0.0),
            "pinch_sz_y": tk.DoubleVar(value=0.5),
            "pinch_gap_y": tk.DoubleVar(value=2.0),
            "pinch_spd_y": tk.DoubleVar(value=1.0),

            "hue": tk.IntVar(value=0),
            "sat": tk.DoubleVar(value=1.0),
            "light": tk.DoubleVar(value=1.0),
            "intensity": tk.DoubleVar(value=1.5),
            "persistence": tk.DoubleVar(value=0.7), 
            "glow": tk.DoubleVar(value=3.0)         
        }

        def add_slider(name, key, f, t, res):
            tk.Label(self.ui_frame, text=name).pack()
            tk.Scale(self.ui_frame, variable=self.vars[key], from_=f, to=t, resolution=res, orient=tk.HORIZONTAL, length=480).pack()

        tk.Label(self.ui_frame, text="--- Geometry ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        add_slider("Morph (0.0 = 3D, 1.0 = Raster)", "morph", 0.0, 1.0, 0.01)
        add_slider("Scanlines", "scanlines", 50, 525, 1)
        
        tk.Label(self.ui_frame, text="--- 2D Raster Transforms ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        add_slider("Translate X", "trans_x", -2.0, 2.0, 0.01)
        add_slider("Translate Y", "trans_y", -2.0, 2.0, 0.01)
        add_slider("Scale X", "scale_x", -3.0, 3.0, 0.01)
        add_slider("Scale Y", "scale_y", -3.0, 3.0, 0.01)
        add_slider("2D Rotation (Radians)", "rot_2d", -3.14, 3.14, 0.01)
        
        add_slider("Pinch X Crush", "crush_x", 0.0, 1.0, 0.01)
        add_slider("Pinch X Stretch", "stretch_x", 0.0, 5.0, 0.01)
        add_slider("Pinch X Vert Size", "pinch_sz_x", 0.05, 3.0, 0.01)
        add_slider("Pinch X Gap Distance", "pinch_gap_x", 0.1, 10.0, 0.1)
        add_slider("Pinch X Travel Speed", "pinch_spd_x", -5.0, 5.0, 0.1)
        
        add_slider("Pinch Y Crush", "crush_y", 0.0, 1.0, 0.01)
        add_slider("Pinch Y Stretch", "stretch_y", 0.0, 5.0, 0.01)
        add_slider("Pinch Y Horiz Size", "pinch_sz_y", 0.05, 3.0, 0.01)
        add_slider("Pinch Y Gap Distance", "pinch_gap_y", 0.1, 10.0, 0.1)
        add_slider("Pinch Y Travel Speed", "pinch_spd_y", -5.0, 5.0, 0.1)

        tk.Label(self.ui_frame, text="--- 3D Lissajous Frequencies ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        
        preset_frame = tk.Frame(self.ui_frame)
        preset_frame.pack(pady=5)
        tk.Button(preset_frame, text="1:1 Circle", command=lambda: self.set_preset(1, 1)).grid(row=0, column=0, padx=2)
        tk.Button(preset_frame, text="2:1 Fig-8", command=lambda: self.set_preset(2, 1)).grid(row=0, column=1, padx=2)
        tk.Button(preset_frame, text="3:2 Knot", command=lambda: self.set_preset(3, 2)).grid(row=0, column=2, padx=2)
        tk.Button(preset_frame, text="4:3 Net", command=lambda: self.set_preset(4, 3)).grid(row=0, column=3, padx=2)
        tk.Button(preset_frame, text="5:4 Web", command=lambda: self.set_preset(5, 4)).grid(row=0, column=4, padx=2)

        add_slider("Freq X", "freq_x", 1, 10, 0.1)
        add_slider("Freq Y", "freq_y", 1, 10, 0.1)
        add_slider("Freq Z (Depth)", "freq_z", 0, 10, 0.1)

        tk.Label(self.ui_frame, text="--- 3D Spin ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        add_slider("Spin Pitch (X)", "spin_x", -5.0, 5.0, 0.1)
        add_slider("Spin Yaw (Y)", "spin_y", -5.0, 5.0, 0.1)
        add_slider("Spin Roll (Z)", "spin_z", -5.0, 5.0, 0.1)
        
        tk.Label(self.ui_frame, text="--- Color & CRT Physics ---", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        add_slider("Hue Shift", "hue", 0, 179, 1)
        add_slider("Saturation", "sat", 0.0, 3.0, 0.1)
        add_slider("Lightness", "light", 0.0, 3.0, 0.1)
        add_slider("Beam Intensity", "intensity", 0.1, 8.0, 0.1)
        add_slider("CRT Persistence", "persistence", 0.0, 0.95, 0.01)
        add_slider("CRT Glow Amount", "glow", 0.0, 10.0, 0.1)

        btn_frame = tk.Frame(self.ui_frame)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="Toggle Cam", command=lambda: setattr(self, 'use_camera', not self.use_camera)).grid(row=0, column=0, padx=2)
        tk.Button(btn_frame, text="Add Key", command=self.add_keyframe).grid(row=0, column=1, padx=2)
        tk.Button(btn_frame, text="Play", command=self.play_timeline).grid(row=0, column=2, padx=2)
        tk.Button(btn_frame, text="Export PNG Seq", command=self.export_sequence, bg="darkred", fg="white").grid(row=0, column=3, padx=2)
        
        self.kf_label = tk.Label(self.ui_frame, text="Keyframes: 0")
        self.kf_label.pack()

    def set_preset(self, fx, fy):
        self.vars["freq_x"].set(fx)
        self.vars["freq_y"].set(fy)
        self.vars["freq_z"].set(max(1.0, fx/2.0)) 

    def add_keyframe(self):
        self.keyframes.append({k: v.get() for k, v in self.vars.items()})
        self.kf_label.config(text=f"Keyframes: {len(self.keyframes)}")

    def play_timeline(self):
        if len(self.keyframes) > 1:
            self.playing = True
            self.play_start_time = time.time()

    def interpolate_keyframes(self, t_elapsed):
        total_time = (len(self.keyframes) - 1) * self.transition_duration
        if t_elapsed >= total_time:
            return False 
        
        idx = int(t_elapsed // self.transition_duration)
        t_norm = (t_elapsed % self.transition_duration) / self.transition_duration
        k1, k2 = self.keyframes[idx], self.keyframes[idx + 1]

        smooth_t = t_norm * t_norm * (3 - 2 * t_norm)
        for key in k1:
            val = k1[key] + (k2[key] - k1[key]) * smooth_t
            self.vars[key].set(val)
        return True

    def apply_colorizer(self, frame, h_shift, s_mult, l_mult):
        if h_shift == 0 and s_mult == 1.0 and l_mult == 1.0: return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * s_mult, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * l_mult, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def compute_frame(self, target_w, target_h, delta_time, buffer_state, is_export=False):
        self.global_time += delta_time
        self.spin_x_acc += self.vars["spin_x"].get() * delta_time
        self.spin_y_acc += self.vars["spin_y"].get() * delta_time
        self.spin_z_acc += self.vars["spin_z"].get() * delta_time
        rx, ry, rz = self.spin_x_acc, self.spin_y_acc, self.spin_z_acc

        if self.use_camera and not is_export:
            ret, frame = self.cap.read()
            if not ret: frame = self.img
        else:
            frame = self.img.copy()
        
        frame = cv2.resize(frame, (self.src_w, self.src_h))
        frame = self.apply_colorizer(frame, self.vars["hue"].get(), self.vars["sat"].get(), self.vars["light"].get())

        colors = (frame.astype(np.float32) / 255.0)

        scale_x = self.vars["scale_x"].get()
        scale_y = self.vars["scale_y"].get()
        # Prevent division by zero inside the JIT compiler
        scale_x = scale_x if abs(scale_x) > 0.001 else 0.001 
        scale_y = scale_y if abs(scale_y) > 0.001 else 0.001

        calculate_raster_numba(
            self.t, self.raster_y, target_w, target_h, self.src_w, self.src_h,
            self.global_time, self.vars["morph"].get(), self.vars["scanlines"].get(), 
            self.vars["intensity"].get(), self.vars["persistence"].get(), self.samples,
            rx, ry, rz, self.vars["freq_x"].get(), self.vars["freq_y"].get(), self.vars["freq_z"].get(),
            self.vars["rot_2d"].get(), self.vars["trans_x"].get(), self.vars["trans_y"].get(), 
            scale_x, scale_y,
            self.vars["pinch_sz_x"].get(), self.vars["pinch_gap_x"].get(), self.vars["crush_x"].get(), self.vars["stretch_x"].get(), self.vars["pinch_spd_x"].get(),
            self.vars["pinch_sz_y"].get(), self.vars["pinch_gap_y"].get(), self.vars["crush_y"].get(), self.vars["stretch_y"].get(), self.vars["pinch_spd_y"].get(),
            buffer_state, colors
        )

        glow_amt = self.vars["glow"].get()
        if glow_amt > 0.1:
            k_size = int(glow_amt * 2) | 1 
            blurred = cv2.GaussianBlur(buffer_state, (k_size, k_size), 0)
            final_canvas = buffer_state + (blurred * 0.5)
        else:
            final_canvas = buffer_state

        return np.clip(final_canvas * 255.0, 0, 255).astype(np.uint8)

    def export_sequence(self):
        if len(self.keyframes) < 2:
            print("Need at least 2 keyframes to export!")
            return

        print("\n--- STARTING OFFLINE EXPORT ---")
        
        # Switch to high-density 400k proxy arrays
        self.t = self.t_export
        self.samples = self.export_samples
        self.raster_y = self.t
        
        fps = 60
        total_transitions = len(self.keyframes) - 1
        total_frames = int(total_transitions * self.transition_duration * fps)
        
        os.makedirs("export", exist_ok=True)
        export_buffer = np.zeros((self.export_h, self.export_w, 3), dtype=np.float32)
        
        self.global_time = 0.0
        self.spin_x_acc = 0.0
        self.spin_y_acc = 0.0
        self.spin_z_acc = 0.0
        fixed_delta = 1.0 / fps

        for i in range(total_frames):
            t_elapsed = i / fps
            self.interpolate_keyframes(t_elapsed)
            
            frame_img = self.compute_frame(self.export_w, self.export_h, fixed_delta, export_buffer, is_export=True)
            
            filename = f"export/frame_{i:05d}.png"
            cv2.imwrite(filename, frame_img)
            
            if i % 10 == 0:
                print(f"Rendered {i}/{total_frames} frames...")

        print("--- EXPORT COMPLETE ---")
        for k, v in self.keyframes[-1].items(): self.vars[k].set(v)
        self.phosphor_buffer.fill(0) 

        # Return to lightweight 40k proxy mode for Tkinter
        self.t = self.t_live
        self.samples = self.live_samples
        self.raster_y = self.t

    def update_loop(self):
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        if self.playing:
            t_elapsed = time.time() - self.play_start_time
            if not self.interpolate_keyframes(t_elapsed):
                self.playing = False
                for k, v in self.keyframes[-1].items(): self.vars[k].set(v)

        if self.phosphor_buffer.shape[:2] != (self.live_h, self.live_w):
            self.phosphor_buffer = np.zeros((self.live_h, self.live_w, 3), dtype=np.float32)
            
        frame_out = self.compute_frame(self.live_w, self.live_h, delta, self.phosphor_buffer)
        
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.vid_label.imgtk = imgtk
        self.vid_label.configure(image=imgtk)

        # Capped 16ms refresh for a smooth 60 FPS viewport
        self.root.after(16, self.update_loop)

    def close_app(self):
        self.cap.release()
        self.root.quit()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.last_time = time.time()
        self.update_loop()
        self.root.mainloop()

if __name__ == "__main__":
    app = ProductionScanimate()
    app.run()