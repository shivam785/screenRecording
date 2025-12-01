import os
import platform
import subprocess
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import webbrowser

import pyautogui
import cv2
import numpy as np
import ffmpeg
import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyttsx3
import speech_recognition as sr
from PIL import Image, ImageTk

APP_NAME = "Smart Screen Recorder Advanced"
PRESET_FILE = Path.home() / ".smart_recorder_presets.json"

recording = False
last_output_file = None
engine = pyttsx3.init()
voice_thread = None

# GUI globals
root = None
status_label = None
elapsed_label = None
progress = None
log_text = None

# Config defaults
cfg = {
    "profile": "web",
    "format": ".mp4",
    # Friendly label now uses multiplier-style wording instead of "fps"
    "screen_speed": "1.0x speed",  # capture speed label (maps internally to FPS)
    "record_mic": True,
    "mic_device": None,
    "record_system": False,
    "system_device": None,
    "record_webcam": True,
    "webcam_device": "0",
    "webcam_position": "Top-right",
    "webcam_size_pct": 20,
    "webcam_border": False,
    "output_dir": str(Path.cwd()),
    "use_duration": False,
    "duration_seconds": 0,
    "speed": 1.0,
    "suppress_dialogs": False
}

def log(msg):
    global log_text
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    if log_text and log_text.winfo_exists():
        log_text.configure(state="normal")
        log_text.insert("end", line)
        log_text.see("end")
        log_text.configure(state="disabled")

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        log(f"TTS error: {e}")

def list_audio_input_devices():
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    inputs = []
    for i, d in enumerate(devices):
        if d.get('max_input_channels', 0) > 0:
            inputs.append((i, d.get('name', '')))
    return inputs

def guess_loopback_devices():
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    candidates = []
    for i, d in enumerate(devices):
        name = d.get('name', '').lower()
        if d.get('max_input_channels', 0) > 0 and ("loopback" in name or "stereo mix" in name or "wave out" in name or "virtual" in name):
            candidates.append((i, d.get('name', '')))
    return candidates

def save_preset(name):
    try:
        if not name:
            return
        presets = {}
        if PRESET_FILE.exists():
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
        presets[name] = cfg.copy()
        with open(PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
        refresh_presets_dropdown()
        log(f"Preset '{name}' saved.")
    except Exception as e:
        log(f"Failed to save preset: {e}")

def load_preset(name):
    try:
        if not PRESET_FILE.exists():
            if not cfg.get("suppress_dialogs"):
                messagebox.showwarning("No presets", "No presets file found.")
            log("No presets file found.")
            return
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            presets = json.load(f)
        if name not in presets:
            if not cfg.get("suppress_dialogs"):
                messagebox.showwarning("Preset not found", f"Preset '{name}' missing.")
            log(f"Preset '{name}' missing.")
            return
        p = presets[name]
        cfg.update(p)
        apply_cfg_to_ui()
        log(f"Preset '{name}' loaded.")
    except Exception as e:
        log(f"Failed to load preset: {e}")

def refresh_presets_dropdown():
    try:
        if not PRESET_FILE.exists():
            presets = {}
        else:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
        names = sorted(presets.keys())
        presets_combobox['values'] = names
    except Exception as e:
        log(f"Error reading presets: {e}")

def apply_cfg_to_ui():
    try:
        profile_var.set("Web (smaller)" if cfg.get("profile","web")=="web" else "Normal (higher)")
        format_var.set(cfg.get("format", ".mp4"))
        screen_speed_var.set(cfg.get("screen_speed", "1.0x speed"))
        mic_var.set(cfg.get("record_mic", True))
        system_var.set(cfg.get("record_system", False))
        microphone_dropdown.set(str(cfg.get("mic_device")) if cfg.get("mic_device") is not None else "")
        system_dropdown.set(str(cfg.get("system_device")) if cfg.get("system_device") is not None else "")
        webcam_var.set(cfg.get("record_webcam", True))
        cam_dropdown.set(cfg.get("webcam_device","0"))
        cam_pos_dropdown.set(cfg.get("webcam_position","Top-right"))
        cam_size_scale.set(cfg.get("webcam_size_pct",20))
        cam_border_var.set(cfg.get("webcam_border", False))
        output_dir_var.set(cfg.get("output_dir", str(Path.cwd())))
        use_duration_var.set(cfg.get("use_duration", False))
        duration_var.set(str(cfg.get("duration_seconds", 0)))
        speed_var.set(str(cfg.get("speed", 1.0)))
        suppress_dialogs_var.set(cfg.get("suppress_dialogs", False))
        on_webcam_toggle()
    except Exception as e:
        log(f"apply_cfg_to_ui error: {e}")

def gather_cfg_from_ui():
    try:
        cfg["profile"] = "web" if "web" in profile_var.get().lower() else "normal"
        cfg["format"] = format_var.get()
        cfg["screen_speed"] = screen_speed_var.get()
        cfg["record_mic"] = bool(mic_var.get())
        mic_dev = microphone_dropdown.get().strip()
        cfg["mic_device"] = int(mic_dev) if mic_dev.isdigit() else None
        cfg["record_system"] = bool(system_var.get())
        sys_dev = system_dropdown.get().strip()
        cfg["system_device"] = int(sys_dev) if sys_dev.isdigit() else None
        cfg["record_webcam"] = bool(webcam_var.get())
        cfg["webcam_device"] = cam_dropdown.get()
        cfg["webcam_position"] = cam_pos_dropdown.get()
        cfg["webcam_size_pct"] = int(cam_size_scale.get())
        cfg["webcam_border"] = bool(cam_border_var.get())
        cfg["output_dir"] = output_dir_var.get().strip() or str(Path.cwd())
        cfg["use_duration"] = bool(use_duration_var.get())
        try:
            cfg["duration_seconds"] = max(0, int(float(duration_var.get())))
        except Exception:
            cfg["duration_seconds"] = 0
        try:
            cfg["speed"] = float(speed_var.get())
            if cfg["speed"] <= 0:
                cfg["speed"] = 1.0
        except Exception:
            cfg["speed"] = 1.0
        cfg["suppress_dialogs"] = bool(suppress_dialogs_var.get())
    except Exception as e:
        log(f"gather_cfg_from_ui error: {e}")

def safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        log(f"Could not create directory {path}: {e}")

def voice_listener_thread():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        log("Microphone not available; voice control disabled.")
        return
    log("Voice control active (start / stop)")
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = recognizer.listen(source, timeout=4)
            cmd = recognizer.recognize_google(audio).lower()
            log(f"Voice heard: {cmd}")
            if "start" in cmd and not recording:
                root.after(0, on_start_clicked)
            if "stop" in cmd and recording:
                root.after(0, on_stop_clicked)
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            log(f"Voice error: {e}")
            time.sleep(1)
            continue

def record_audio_to_file(audio_filename, mic_device_index, system_device_index, use_mic, use_system, stop_flag):
    fs = 44100
    channels = 2
    frames = []
    def callback(indata, frames_count, time_info, status):
        if status:
            log(f"Audio status: {status}")
        frames.append(indata.copy())
        if stop_flag["stop"]:
            raise sd.CallbackStop
    if not use_mic and not use_system:
        log("Audio disabled.")
        return
    try:
        device = mic_device_index if use_mic and mic_device_index is not None else None
        with sd.InputStream(samplerate=fs, channels=channels, device=device, callback=callback):
            while not stop_flag["stop"]:
                sd.sleep(100)
    except sd.CallbackStop:
        pass
    except Exception as e:
        log(f"Audio capture error: {e}")
    if frames:
        audio = np.concatenate(frames, axis=0)
        if audio.dtype != np.int16:
            try:
                clipped = np.clip(audio, -1.0, 1.0)
                int_audio = (clipped * 32767).astype(np.int16)
            except Exception:
                int_audio = audio.astype(np.int16, copy=False)
        else:
            int_audio = audio
        try:
            write(audio_filename, fs, int_audio)
            log(f"Saved audio: {audio_filename}")
        except Exception as e:
            log(f"Failed writing audio file: {e}")
    else:
        log("No audio frames captured.")

def build_atempo_factors(speed):
    if speed <= 0:
        return [1.0]
    factors = []
    s = speed
    while s > 2.0:
        factors.append(2.0)
        s /= 2.0
    while s < 0.5:
        factors.append(0.5)
        s /= 0.5
    factors.append(round(s, 5))
    return factors

def ffmpeg_merge(video_file, audio_file, out_file, profile, speed=1.0, suppress_dialogs=False):
    try:
        input_video = ffmpeg.input(video_file)
        video_stream = input_video
        audio_stream = None
        if speed != 1.0:
            video_stream = ffmpeg.filter(input_video, "setpts", f"PTS/{float(speed)}")
        if audio_file and os.path.exists(audio_file):
            input_audio = ffmpeg.input(audio_file)
            audio_stream = input_audio
            if speed != 1.0:
                factors = build_atempo_factors(speed)
                for f in factors:
                    audio_stream = audio_stream.filter("atempo", f)
        if audio_stream is not None:
            out = ffmpeg.output(video_stream, audio_stream, out_file, vcodec='libx264', acodec='aac',
                                crf=(26 if profile=="web" else 20), preset=('veryfast' if profile=="web" else 'faster'),
                                movflags='+faststart', strict='experimental')
        else:
            out = ffmpeg.output(video_stream, out_file, vcodec='libx264',
                                crf=(26 if profile=="web" else 20), preset=('veryfast' if profile=="web" else 'faster'),
                                movflags='+faststart')
        out.run(overwrite_output=True)
        log(f"FFmpeg finished: {out_file}")
        return True, ""
    except ffmpeg.Error as e:
        try:
            err = e.stderr.decode()
        except Exception:
            err = str(e)
        log(f"FFmpeg error: {err}")
        return False, err
    except Exception as e:
        log(f"FFmpeg unexpected error: {e}")
        return False, str(e)

def record_worker(video_path, audio_path, final_path, cfg_local, stop_flag):
    global recording, last_output_file
    screen_w, screen_h = pyautogui.size()
    # Map user-friendly screen_speed (multiplier-style labels) to capture FPS
    speed_map = {
        "0.5x speed": 10,   # lower FPS -> more "slow" capture, shown as 0.5x
        "1.0x speed": 20,   # normal
        "1.5x speed": 30,   # faster capture
        "1.75x speed": 45,  # very fast
        "2.0x speed": 60    # ultra
    }
    fps = speed_map.get(cfg_local.get("screen_speed"), 20)
    frame_delay = 1.0 / fps if fps > 0 else 1.0/20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (screen_w, screen_h))
    cam = None
    if cfg_local["record_webcam"]:
        try:
            cam = cv2.VideoCapture(int(cfg_local["webcam_device"]),
                                   cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
            if cam and cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                try:
                    cam.release()
                except Exception:
                    pass
                cam = None
                log("Webcam not available; continuing without webcam.")
        except Exception as e:
            log(f"Webcam error: {e}")
            cam = None
    audio_thread = None
    audio_stop = {"stop": False}
    if cfg_local["record_mic"] or cfg_local["record_system"]:
        audio_thread = threading.Thread(target=record_audio_to_file,
                                        args=(audio_path, cfg_local["mic_device"], cfg_local["system_device"],
                                              cfg_local["record_mic"], cfg_local["record_system"], audio_stop),
                                        daemon=True)
        audio_thread.start()
    start_t = time.time()
    log("Screen recording started.")
    update_status("Recording...", "red")
    start_progress()
    elapsed_last = -1
    try:
        while not stop_flag["stop"]:
            loop_start = time.time()
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            mx, my = pyautogui.position()
            cv2.circle(frame, (mx, my), 5, (0,255,255), -1)
            if cam is not None:
                ret, cam_frame = cam.read()
                if ret and cam_frame is not None:
                    try:
                        ch, cw = cam_frame.shape[:2]
                        overlay_w = max(80, int(screen_w * (cfg_local["webcam_size_pct"] / 100.0)))
                        overlay_h = int(overlay_w * (ch / cw))
                        small = cv2.resize(cam_frame, (overlay_w, overlay_h))
                        pos = cfg_local["webcam_position"]
                        if pos == "Top-left":
                            x0, y0 = 10, 10
                        elif pos == "Top-right":
                            x0, y0 = screen_w - overlay_w - 10, 10
                        elif pos == "Bottom-left":
                            x0, y0 = 10, screen_h - overlay_h - 40
                        else:
                            x0, y0 = screen_w - overlay_w - 10, screen_h - overlay_h - 40
                        try:
                            frame[y0:y0+overlay_h, x0:x0+overlay_w] = small
                            if cfg_local.get("webcam_border"):
                                cv2.rectangle(frame, (x0,y0),(x0+overlay_w,y0+overlay_h),(0,0,0),2)
                        except Exception:
                            pass
                    except Exception:
                        pass
            elapsed = int(time.time() - start_t)
            if cfg_local.get("use_duration") and cfg_local.get("duration_seconds", 0) > 0 and elapsed >= cfg_local.get("duration_seconds"):
                log("Duration reached; stopping recording.")
                break
            if elapsed != elapsed_last:
                elapsed_last = elapsed
                try:
                    root.after(0, lambda v=elapsed: elapsed_label.config(text=f"Elapsed: {v}s"))
                except Exception:
                    pass
            cv2.putText(frame, f"Elapsed: {elapsed}s", (10, screen_h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            out.write(frame)
            elapsed_loop = time.time() - loop_start
            to_sleep = frame_delay - elapsed_loop
            if to_sleep > 0:
                time.sleep(to_sleep)
    finally:
        try:
            out.release()
        except Exception:
            pass
        if cam:
            try:
                cam.release()
            except Exception:
                pass
        if audio_thread:
            audio_stop["stop"] = True
            try:
                audio_thread.join(timeout=3)
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
    update_status("Finalizing...", "orange")
    stop_progress()
    log("Finalizing: merging / encoding")
    ok, err = ffmpeg_merge(video_path, audio_path if os.path.exists(audio_path) else None, final_path, cfg_local["profile"], speed=cfg_local.get("speed", 1.0), suppress_dialogs=cfg_local.get("suppress_dialogs", False))
    if not ok:
        log(f"Finalization error: {err}")
        if not cfg_local.get("suppress_dialogs", False):
            messagebox.showerror("FFmpeg error", f"Failed to finalize recording.\n{err}")
        update_status("Idle", "green")
        return
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass
    last_output_file = os.path.abspath(final_path)
    log(f"Saved: {final_path}")
    try:
        if not cfg_local.get("suppress_dialogs", False):
            messagebox.showinfo("Done", f"Recording saved as:\n{final_path}")
    except Exception:
        pass
    update_status("Idle", "green")

def start_progress():
    try:
        progress.start(10)
    except Exception:
        pass

def stop_progress():
    try:
        progress.stop()
    except Exception:
        pass

def update_status(text, color="white"):
    if status_label and status_label.winfo_exists():
        status_label.config(text=text, fg=color)

def on_start_clicked(event=None):
    global recording
    if recording:
        return
    gather_cfg_from_ui()
    safe_makedirs(cfg["output_dir"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tmp_video = os.path.join(cfg["output_dir"], f"tmp_video_{timestamp}.mp4")
    tmp_audio = os.path.join(cfg["output_dir"], f"tmp_audio_{timestamp}.wav")
    final_out = os.path.join(cfg["output_dir"], f"Recording_{timestamp}{cfg['format']}")
    stop_flag = {"stop": False}
    recording = True
    update_status("Preparing...", "orange")
    log("Starting recording...")
    worker = threading.Thread(target=record_worker, args=(tmp_video, tmp_audio, final_out, cfg.copy(), stop_flag), daemon=True)
    worker.start()
    def stop_wait():
        global recording
        if cfg.get("use_duration") and cfg.get("duration_seconds", 0) > 0:
            pass
        while recording:
            time.sleep(0.1)
        stop_flag["stop"] = True
    threading.Thread(target=stop_wait, daemon=True).start()

def on_stop_clicked(event=None):
    global recording
    if not recording:
        return
    log("Stop requested...")
    recording = False

def on_browse_output_dir():
    d = filedialog.askdirectory(initialdir=cfg.get("output_dir", str(Path.cwd())))
    if d:
        output_dir_var.set(d)

def on_refresh_devices():
    devices = list_audio_input_devices()
    microphone_dropdown['values'] = [str(i) for i, n in devices]
    if devices:
        microphone_dropdown.set(str(devices[0][0]))
    sys_candidates = guess_loopback_devices()
    system_dropdown['values'] = [str(i) for i, n in sys_candidates] if sys_candidates else []
    if sys_candidates:
        system_dropdown.set(str(sys_candidates[0][0]))

def on_webcam_toggle():
    enabled = webcam_var.get()
    state = "normal" if enabled else "disabled"
    cam_dropdown.config(state=state)
    cam_pos_dropdown.config(state=state)
    cam_size_scale.config(state=state)
    cam_border_checkbox.config(state=state)

def create_ui():
    global root, status_label, elapsed_label, progress, log_text
    global profile_var, format_var, screen_speed_var
    global mic_var, microphone_dropdown, system_var, system_dropdown
    global webcam_var, cam_dropdown, cam_pos_dropdown, cam_size_scale, cam_border_var, cam_border_checkbox
    global output_dir_var, presets_combobox, use_duration_var, duration_var, speed_var, suppress_dialogs_var

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("780x680")
    root.configure(bg="#111827")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    header = tk.Frame(root, bg="#0b1220")
    header.pack(fill="x")
    tk.Label(header, text=APP_NAME, font=("Segoe UI", 16, "bold"), bg="#0b1220", fg="white").pack(side="left", padx=10, pady=8)

    main = ttk.Notebook(root)
    main.pack(fill="both", expand=True, padx=12, pady=10)

    tab_general = ttk.Frame(main)
    tab_audio = ttk.Frame(main)
    tab_webcam = ttk.Frame(main)
    tab_adv = ttk.Frame(main)

    main.add(tab_general, text="General")
    main.add(tab_audio, text="Audio")
    main.add(tab_webcam, text="Webcam")
    main.add(tab_adv, text="Advanced")

    # General tab
    gp = ttk.LabelFrame(tab_general, text="Output & Profile")
    gp.pack(fill="x", padx=10, pady=8)
    profile_var = tk.StringVar(value="Web (smaller)")
    profile_cb = ttk.Combobox(gp, textvariable=profile_var, values=["Normal (higher)", "Web (smaller)"], state="readonly", width=24)
    profile_cb.grid(row=0, column=0, padx=8, pady=8)
    format_var = tk.StringVar(value=".mp4")
    format_cb = ttk.Combobox(gp, textvariable=format_var, values=[".mp4", ".mkv", ".mov"], state="readonly", width=12)
    format_cb.grid(row=0, column=1, padx=8, pady=8)

    # --- Capture Speed selector (multiplier-style labels) ---
    ttk.Label(gp, text="Capture speed (affects capture FPS):").grid(row=0, column=2, padx=(16,4))
    screen_speed_var = tk.StringVar(value=cfg["screen_speed"])
    screen_speed_dropdown = ttk.Combobox(
        gp,
        textvariable=screen_speed_var,
        values=[
            "0.5x speed",
            "1.0x speed",
            "1.5x speed",
            "1.75x speed",
            "2.0x speed"
        ],
        state="readonly",
        width=20
    )
    screen_speed_dropdown.grid(row=0, column=3, padx=6)

    ttk.Label(gp, text="Output folder:").grid(row=1, column=0, padx=8, pady=(4,8), sticky="w")
    output_dir_var = tk.StringVar(value=cfg["output_dir"])
    tk.Entry(gp, textvariable=output_dir_var, width=56).grid(row=1, column=1, columnspan=3, padx=8, pady=(4,8))
    ttk.Button(gp, text="Browse", command=on_browse_output_dir).grid(row=1, column=4, padx=6, pady=(4,8))

    # Duration controls
    dur_frame = ttk.LabelFrame(tab_general, text="Duration (optional)")
    dur_frame.pack(fill="x", padx=10, pady=6)
    use_duration_var = tk.BooleanVar(value=cfg["use_duration"])
    tk.Checkbutton(dur_frame, text="Record for fixed duration (seconds)", variable=use_duration_var, bg="#111827", fg="white", selectcolor="#111827").grid(row=0, column=0, padx=8, pady=6, sticky="w")
    duration_var = tk.StringVar(value=str(cfg["duration_seconds"]))
    tk.Entry(dur_frame, textvariable=duration_var, width=12).grid(row=0, column=1, padx=8)
    tk.Label(dur_frame, text="seconds").grid(row=0, column=2, sticky="w")

    ctrl_frame = ttk.LabelFrame(tab_general, text="Controls")
    ctrl_frame.pack(fill="x", padx=10, pady=8)
    start_btn = ttk.Button(ctrl_frame, text="● Start Recording (Ctrl+R)", command=on_start_clicked)
    start_btn.grid(row=0, column=0, padx=8, pady=8)
    stop_btn = ttk.Button(ctrl_frame, text="■ Stop Recording (Ctrl+S)", command=on_stop_clicked)
    stop_btn.grid(row=0, column=1, padx=8, pady=8)
    progress = ttk.Progressbar(ctrl_frame, mode="indeterminate", length=220)
    progress.grid(row=0, column=2, padx=8)
    status_label = tk.Label(ctrl_frame, text="Idle", bg="#111827", fg="#22c55e", font=("Segoe UI", 10))
    status_label.grid(row=0, column=3, padx=8)
    elapsed_label = tk.Label(ctrl_frame, text="Elapsed: 0s", bg="#111827", fg="white", font=("Segoe UI", 10))
    elapsed_label.grid(row=0, column=4, padx=8)

    presets_frame = ttk.LabelFrame(tab_general, text="Presets")
    presets_frame.pack(fill="x", padx=10, pady=6)
    presets_combobox = ttk.Combobox(presets_frame, values=[], state="readonly", width=28)
    presets_combobox.grid(row=0, column=0, padx=8, pady=8)
    ttk.Button(presets_frame, text="Load", command=lambda: load_preset(presets_combobox.get())).grid(row=0, column=1, padx=6)
    ttk.Button(presets_frame, text="Refresh", command=refresh_presets_dropdown).grid(row=0, column=2, padx=6)
    ttk.Button(presets_frame, text="Save As...", command=lambda: save_preset(simple_input_dialog("Preset name"))).grid(row=0, column=3, padx=6)

    # Audio tab
    a1 = ttk.LabelFrame(tab_audio, text="Mic / System Audio")
    a1.pack(fill="x", padx=10, pady=8)
    mic_var = tk.BooleanVar(value=cfg["record_mic"])
    tk.Checkbutton(a1, text="Record Microphone", variable=mic_var, bg="#111827", fg="white", selectcolor="#111827").grid(row=0, column=0, padx=8, pady=6, sticky="w")
    ttk.Button(a1, text="Refresh devices", command=on_refresh_devices).grid(row=0, column=1, padx=8)
    microphone_dropdown = ttk.Combobox(a1, values=[str(i) for i, n in list_audio_input_devices()], width=20)
    microphone_dropdown.grid(row=1, column=0, padx=8, pady=6, sticky="w")
    tk.Label(a1, text="Mic device index").grid(row=1, column=1, sticky="w")

    system_var = tk.BooleanVar(value=cfg["record_system"])
    tk.Checkbutton(a1, text="Record System Audio (loopback)", variable=system_var, bg="#111827", fg="white", selectcolor="#111827").grid(row=2, column=0, padx=8, pady=6, sticky="w")
    system_dropdown = ttk.Combobox(a1, values=[str(i) for i, n in guess_loopback_devices()], width=20)
    system_dropdown.grid(row=3, column=0, padx=8, pady=6, sticky="w")
    tk.Label(a1, text="System device index (if available)").grid(row=3, column=1, sticky="w")

    # Webcam tab
    w1 = ttk.LabelFrame(tab_webcam, text="Webcam overlay")
    w1.pack(fill="x", padx=10, pady=8)
    webcam_var = tk.BooleanVar(value=cfg["record_webcam"])
    tk.Checkbutton(w1, text="Enable webcam overlay", variable=webcam_var, bg="#111827", fg="white", selectcolor="#111827", command=on_webcam_toggle).grid(row=0, column=0, padx=8, pady=6, sticky="w")
    tk.Label(w1, text="Camera device index").grid(row=1, column=0, sticky="w", padx=8)
    cams = list_webcams_ui() if 'list_webcams_ui' in globals() else []
    if not cams:
        cams = ["0"]
    cam_dropdown = ttk.Combobox(w1, values=cams, width=12)
    cam_dropdown.set(cams[0] if cams else "0")
    cam_dropdown.grid(row=1, column=1, padx=8)
    preview_btn = ttk.Button(w1, text="Preview camera", command=start_preview)
    preview_btn.grid(row=1, column=2, padx=6)
    tk.Label(w1, text="Position").grid(row=2, column=0, padx=8, sticky="w")
    cam_pos_dropdown = ttk.Combobox(w1, values=["Top-left","Top-right","Bottom-left","Bottom-right"], width=12)
    cam_pos_dropdown.set("Top-right")
    cam_pos_dropdown.grid(row=2, column=1, padx=8)
    tk.Label(w1, text="Size (% of width)").grid(row=3, column=0, padx=8, sticky="w")
    cam_size_scale = tk.Scale(w1, from_=10, to=40, orient="horizontal")
    cam_size_scale.set(20)
    cam_size_scale.grid(row=3, column=1, padx=8, pady=6)
    cam_border_var = tk.BooleanVar(value=False)
    cam_border_checkbox = tk.Checkbutton(w1, text="Draw border around webcam", variable=cam_border_var, bg="#111827", fg="white", selectcolor="#111827")
    cam_border_checkbox.grid(row=4, column=0, padx=8, pady=6, sticky="w")

    # Advanced tab
    adv = ttk.LabelFrame(tab_adv, text="Advanced / Logs")
    adv.pack(fill="both", expand=True, padx=10, pady=8)

    adv_top = tk.Frame(adv)
    adv_top.pack(fill="x", padx=6, pady=6)
    tk.Label(adv_top, text="Playback speed:").grid(row=0, column=0, padx=6, sticky="w")
    speed_var = tk.StringVar(value=str(cfg["speed"]))
    speed_cb = ttk.Combobox(adv_top, textvariable=speed_var, values=["0.5","0.75","1.0","1.25","1.5","2.0","3.0","4.0"], width=8, state="readonly")
    speed_cb.grid(row=0, column=1, padx=6)
    suppress_dialogs_var = tk.BooleanVar(value=cfg["suppress_dialogs"])
    tk.Checkbutton(adv_top, text="Suppress dialogs while recording (log only)", variable=suppress_dialogs_var, bg="#111827", fg="white", selectcolor="#111827").grid(row=0, column=2, padx=16)

    tk.Label(adv, text="Application Log").pack(anchor="w", padx=8)
    log_frame = tk.Frame(adv)
    log_frame.pack(fill="both", expand=True, padx=8, pady=6)
    scrollbar = tk.Scrollbar(log_frame)
    scrollbar.pack(side="right", fill="y")
    global log_text
    log_text = tk.Text(log_frame, height=12, state="disabled", yscrollcommand=scrollbar.set)
    log_text.pack(fill="both", expand=True)
    scrollbar.config(command=log_text.yview)

    # Bind hotkeys
    root.bind("<Control-r>", lambda e: on_start_clicked())
    root.bind("<Control-s>", lambda e: on_stop_clicked())

    refresh_presets_dropdown()
    apply_cfg_to_ui()
    on_refresh_devices()
    on_webcam_toggle()

    bottom = tk.Frame(root, bg="#0b1220")
    bottom.pack(fill="x")
    ttk.Button(bottom, text="Open Last Recording", command=lambda: open_file(last_output_file) if last_output_file else messagebox.showinfo("No recording","No recording yet.")).pack(side="left", padx=8, pady=6)
    ttk.Button(bottom, text="Open Output Folder", command=lambda: open_file(cfg["output_dir"])).pack(side="left", padx=8, pady=6)

def list_webcams_ui(max_idx=5):
    found = []
    for i in range(max_idx+1):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
            if cap is None or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            found.append(str(i))
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            continue
    return found if found else ["0"]

# Preview window
preview_cap = None
preview_top = None
def start_preview():
    global preview_cap, preview_top
    try:
        idx = int(cam_dropdown.get() if cam_dropdown.get().isdigit() else 0)
    except Exception:
        idx = 0
    try:
        preview_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
    except Exception as e:
        messagebox.showwarning("Preview", f"Could not open camera {idx}: {e}")
        preview_cap = None
        return
    if not preview_cap or not preview_cap.isOpened():
        messagebox.showwarning("Preview", f"Could not open camera {idx}")
        try:
            if preview_cap:
                preview_cap.release()
        except Exception:
            pass
        preview_cap = None
        return
    preview_top = tk.Toplevel(root)
    preview_top.title("Webcam Preview")
    preview_top.geometry("640x480")
    lbl = tk.Label(preview_top)
    lbl.pack(fill="both", expand=True)
    def loop():
        if not preview_top or not preview_top.winfo_exists():
            stop_preview()
            return
        ret, frame = preview_cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((640,480))
            imgtk = ImageTk.PhotoImage(img)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk)
        preview_top.after(30, loop)
    loop()

def stop_preview():
    global preview_cap, preview_top
    try:
        if preview_cap:
            preview_cap.release()
    except Exception:
        pass
    try:
        if preview_top and preview_top.winfo_exists():
            preview_top.destroy()
    except Exception:
        pass
    preview_cap = None
    preview_top = None

def simple_input_dialog(prompt):
    d = tk.Toplevel(root)
    d.title(prompt)
    tk.Label(d, text=prompt).pack(padx=8, pady=6)
    v = tk.StringVar()
    tk.Entry(d, textvariable=v, width=36).pack(padx=8, pady=6)
    result = {"val": None}
    def ok():
        result["val"] = v.get().strip()
        d.destroy()
    def cancel():
        d.destroy()
    tk.Button(d, text="OK", command=ok).pack(side="left", padx=8, pady=8)
    tk.Button(d, text="Cancel", command=cancel).pack(side="right", padx=8, pady=8)
    d.grab_set()
    root.wait_window(d)
    return result["val"]

def open_file(path):
    if not path:
        return
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        log(f"open_file error: {e}")

def start_voice_listener():
    global voice_thread
    voice_thread = threading.Thread(target=voice_listener_thread, daemon=True)
    voice_thread.start()

if __name__ == "__main__":
    create_ui()
    start_voice_listener()
    root.mainloop()
