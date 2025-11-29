import pyautogui
import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import time
import os
import platform
import subprocess
from datetime import datetime
import ffmpeg
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pyttsx3
import speech_recognition as sr
import webbrowser
from pathlib import Path

recording = False
voice_thread = None
profile_choice = "normal"
last_output_file = None

record_mic = True
record_system = False
record_webcam = True

output_extension = ".mp4"

root = None
start_btn = None
stop_btn = None
status_label = None
format_dropdown = None
ext_dropdown = None
mic_var = None
system_var = None
webcam_var = None

engine = pyttsx3.init()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[TTS WARN]", e)

def open_file(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print("[OPEN WARN]", e)

def voice_listener():
    global recording, root
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        print("[VOICE] Microphone not available — voice control disabled.")
        return
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3)
            cmd = recognizer.recognize_google(audio).lower()
            print("[Voice Heard]", cmd)
            if "start" in cmd and not recording:
                try:
                    speak("Starting recording")
                except Exception:
                    pass
                if root and root.winfo_exists():
                    root.after(0, start_recording)
            if "stop" in cmd and recording:
                try:
                    speak("Stopping recording")
                except Exception:
                    pass
                if root and root.winfo_exists():
                    root.after(0, stop_recording)
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print("[VOICE ERROR]", e)
            continue

def record_audio(audio_filename, use_mic: bool, use_system: bool):
    global recording
    fs = 44100
    channels = 2
    frames = []
    if not use_mic and not use_system:
        print("[AUDIO] Disabled")
        return
    def callback(indata, frame_count, time_info, status):
        if status:
            print("[AUDIO STATUS]", status)
        frames.append(indata.copy())
        if not recording:
            raise sd.CallbackStop
    try:
        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            while recording:
                sd.sleep(100)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print("[AUDIO ERROR]", e)
    if frames:
        audio = np.concatenate(frames, axis=0)
        if audio.dtype != np.int16:
            try:
                scaled = np.clip(audio, -1.0, 1.0)
                int_audio = (scaled * 32767).astype(np.int16)
            except Exception:
                int_audio = audio.astype(np.int16)
        else:
            int_audio = audio
        try:
            write(audio_filename, fs, int_audio)
            print("[AUDIO] Saved", audio_filename)
        except Exception as e:
            print("[AUDIO WRITE ERROR]", e)
    else:
        print("[AUDIO] No frames captured")

def create_and_open_player(video_path):
    video_path = Path(video_path).absolute()
    if not video_path.exists():
        messagebox.showerror("Error", "Video file does not exist.")
        return
    ext = video_path.suffix.lower().lstrip(".")
    mime_map = {"mp4":"video/mp4","mov":"video/mp4","mkv":"video/mp4","avi":"video/x-msvideo"}
    mime_type = mime_map.get(ext,"video/mp4")
    html_path = video_path.with_name(video_path.stem + "_player.html")
    html_content = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Player</title>
<style>body{{background:#111827;color:#e5e7eb;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0;font-family:system-ui;}}video{{max-width:90%;max-height:80vh;border-radius:12px;}}</style>
</head><body><h1>🎥 Recording</h1><video controls autoplay><source src="{video_path.name}" type="{mime_type}">Your browser does not support video.</video></body></html>"""
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        webbrowser.open(html_path.as_uri())
    except Exception as e:
        print("[PLAYER ERROR]", e)

def record_screen(video_filename, audio_filename, final_output, profile, use_mic, use_system, use_webcam):
    global recording, last_output_file
    screen_width, screen_height = pyautogui.size()
    resolution = (screen_width, screen_height)
    fps = 20.0
    frame_delay = 1.0 / fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, resolution)
    cam = None
    if use_webcam:
        try:
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system()=="Windows" else 0)
            if not cam.isOpened():
                print("[WEBCAM] Could not open webcam; continuing without webcam.")
                cam.release()
                cam = None
            else:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        except Exception as e:
            print("[WEBCAM ERROR]", e)
            cam = None
    start_time = time.time()
    audio_thread = None
    if use_mic or use_system:
        audio_thread = threading.Thread(target=record_audio, args=(audio_filename, use_mic, use_system), daemon=True)
        audio_thread.start()
    update_status("Recording...", "red")
    update_buttons(True)
    try:
        while recording:
            loop_start = time.time()
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cursor_x, cursor_y = pyautogui.position()
            cv2.circle(frame, (cursor_x, cursor_y), 5, (0,255,255), -1)
            if cam is not None:
                ret, webcam_frame = cam.read()
                if ret:
                    try:
                        webcam_small = cv2.resize(webcam_frame, (320,240))
                        frame[10:10+240, 10:10+320] = webcam_small
                    except Exception:
                        pass
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"Elapsed: {elapsed}s", (10, resolution[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            out.write(frame)
            elapsed_loop = time.time() - loop_start
            sleep_time = frame_delay - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        out.release()
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
    print("[FINALIZING] Merging/transcoding...")
    try:
        if (use_mic or use_system) and os.path.exists(audio_filename):
            input_video = ffmpeg.input(video_filename)
            input_audio = ffmpeg.input(audio_filename)
            if profile == "web":
                stream = ffmpeg.filter(input_video, 'scale', 1280, -2)
            else:
                stream = input_video
            out_stream = ffmpeg.output(stream, input_audio, final_output, vcodec='libx264', acodec='aac',
                                       crf=(26 if profile=="web" else 20), preset=('veryfast' if profile=="web" else 'faster'),
                                       movflags='+faststart', strict='experimental')
            out_stream.run(overwrite_output=True)
        else:
            input_video = ffmpeg.input(video_filename)
            out_stream = ffmpeg.output(input_video, final_output, vcodec='libx264',
                                       crf=(26 if profile=="web" else 20),
                                       preset=('veryfast' if profile=="web" else 'faster'),
                                       movflags='+faststart')
            out_stream.run(overwrite_output=True)
    except ffmpeg.Error as e:
        print("[FFMPEG ERROR]", e.stderr.decode() if hasattr(e, "stderr") else e)
        messagebox.showerror("Error", f"FFmpeg failed:\n{e}")
        update_status("Idle", "green")
        update_buttons(False)
        try:
            if os.path.exists(video_filename):
                os.remove(video_filename)
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
        except Exception:
            pass
        return
    except Exception as e:
        print("[FINALIZE ERROR]", e)
        messagebox.showerror("Error", f"Finalization failed:\n{e}")
        update_status("Idle", "green")
        update_buttons(False)
        return
    finally:
        try:
            if os.path.exists(video_filename):
                os.remove(video_filename)
        except Exception:
            pass
        try:
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
        except Exception:
            pass
    last_output_file = os.path.abspath(final_output)
    print("[SAVED]", last_output_file)
    try:
        messagebox.showinfo("Done", f"Recording saved as:\n{final_output}")
    except Exception:
        pass
    update_status("Idle", "green")
    update_buttons(False)
    try:
        create_and_open_player(last_output_file)
    except Exception as e:
        print("[PLAYER WARN]", e)

def update_status(text, color="white"):
    global status_label
    if status_label and status_label.winfo_exists():
        status_label.after(0, lambda: status_label.config(text=text, fg=color))

def update_buttons(is_recording: bool):
    global start_btn, stop_btn
    if start_btn and stop_btn and start_btn.winfo_exists() and stop_btn.winfo_exists():
        def _upd():
            if is_recording:
                start_btn.config(state="disabled")
                stop_btn.config(state="normal")
            else:
                start_btn.config(state="normal")
                stop_btn.config(state="disabled")
        start_btn.after(0, _upd)

def start_recording(event=None):
    global recording, profile_choice, record_mic, record_system, record_webcam, output_extension
    if recording:
        return
    if not format_dropdown or not ext_dropdown:
        return
    label = format_dropdown.get().lower()
    profile_choice = "web" if "web" in label else "normal"
    record_mic = bool(mic_var.get())
    record_system = bool(system_var.get())
    record_webcam = bool(webcam_var.get())
    ext_label = ext_dropdown.get().lower()
    output_extension = ".mkv" if "mkv" in ext_label else (".mov" if "mov" in ext_label else ".mp4")
    recording = True
    update_status("Preparing...", "orange")
    update_buttons(True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = f"temp_video_{timestamp}.mp4"
    audio_filename = f"temp_audio_{timestamp}.wav"
    final_output = f"Recording_{timestamp}{output_extension}"
    threading.Thread(target=record_screen, args=(video_filename, audio_filename, final_output, profile_choice, record_mic, record_system, record_webcam), daemon=True).start()

def stop_recording(event=None):
    global recording
    if not recording:
        return
    recording = False
    update_status("Stopping...", "orange")
    update_buttons(False)

def play_in_browser():
    global last_output_file
    if last_output_file and os.path.exists(last_output_file):
        try:
            create_and_open_player(last_output_file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open:\n{e}")
    else:
        messagebox.showwarning("No Recording", "No recording found yet. Please record first.")

def launch_gui():
    global root, start_btn, stop_btn, status_label, format_dropdown, ext_dropdown
    global mic_var, system_var, webcam_var, voice_thread
    root = tk.Tk()
    root.title("Smart Screen Recorder")
    root.resizable(False, False)
    window_width = 440
    window_height = 380
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    x = int((screen_width - window_width) / 2)
    y = 10
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.attributes("-topmost", True)
    root.configure(bg="#111827")
    title_label = tk.Label(root, text="🎥 Smart Screen Recorder", font=("Segoe UI", 14, "bold"), bg="#111827", fg="white")
    title_label.pack(pady=(10,4))
    subtitle = tk.Label(root, text="Say 'start' / 'stop' • Ctrl+R = Start • Ctrl+S = Stop", font=("Segoe UI", 9), bg="#111827", fg="#9ca3af")
    subtitle.pack(pady=(0,10))
    options_frame = tk.LabelFrame(root, text=" Options ", bg="#111827", fg="#e5e7eb", font=("Segoe UI",9,"bold"), bd=1, relief="groove", padx=10, pady=8)
    options_frame.pack(padx=10, pady=5, fill="x")
    fmt_label = tk.Label(options_frame, text="Quality profile:", bg="#111827", fg="white", font=("Segoe UI",9))
    fmt_label.grid(row=0, column=0, padx=(0,5), pady=2, sticky="w")
    format_dropdown = ttk.Combobox(options_frame, values=["Normal MP4 (higher quality)","Web MP4 (for websites, smaller size)"], state="readonly", width=32)
    format_dropdown.current(1)
    format_dropdown.grid(row=0, column=1, pady=2, sticky="w")
    ext_label = tk.Label(options_frame, text="Output format:", bg="#111827", fg="white", font=("Segoe UI",9))
    ext_label.grid(row=1, column=0, padx=(0,5), pady=2, sticky="w")
    ext_dropdown = ttk.Combobox(options_frame, values=["MP4 (.mp4, recommended)","MKV (.mkv)","MOV (.mov)"], state="readonly", width=32)
    ext_dropdown.current(0)
    ext_dropdown.grid(row=1, column=1, pady=2, sticky="w")
    av_frame = tk.LabelFrame(root, text=" Audio & Video ", bg="#111827", fg="#e5e7eb", font=("Segoe UI",9,"bold"), bd=1, relief="groove", padx=10, pady=8)
    av_frame.pack(padx=10, pady=5, fill="x")
    mic_var = tk.BooleanVar(value=True)
    system_var = tk.BooleanVar(value=False)
    webcam_var = tk.BooleanVar(value=True)
    mic_chk = tk.Checkbutton(av_frame, text="Record Microphone", variable=mic_var, bg="#111827", fg="white", activebackground="#111827", activeforeground="white", selectcolor="#1f2937", font=("Segoe UI",9))
    mic_chk.grid(row=0, column=0, sticky="w")
    system_chk = tk.Checkbutton(av_frame, text="Record System Sound (requires loopback device)", variable=system_var, bg="#111827", fg="white", activebackground="#111827", activeforeground="white", selectcolor="#1f2937", font=("Segoe UI",9), wraplength=260, justify="left")
    system_chk.grid(row=1, column=0, sticky="w", pady=(2,0))
    webcam_chk = tk.Checkbutton(av_frame, text="Show Webcam Overlay", variable=webcam_var, bg="#111827", fg="white", activebackground="#111827", activeforeground="white", selectcolor="#1f2937", font=("Segoe UI",9))
    webcam_chk.grid(row=2, column=0, sticky="w", pady=(4,0))
    btn_frame = tk.Frame(root, bg="#111827")
    btn_frame.pack(pady=12)
    start_btn = tk.Button(btn_frame, text="● Start Recording", font=("Segoe UI",11,"bold"), width=18, command=start_recording, bg="#22c55e", fg="white", activebackground="#16a34a", activeforeground="white", relief="flat", padx=5, pady=8)
    start_btn.grid(row=0, column=0, padx=8)
    stop_btn = tk.Button(btn_frame, text="■ Stop", font=("Segoe UI",11,"bold"), width=10, command=stop_recording, bg="#ef4444", fg="white", activebackground="#b91c1c", activeforeground="white", relief="flat", padx=5, pady=8, state="disabled")
    stop_btn.grid(row=0, column=1, padx=8)
    web_btn = tk.Button(root, text="▶ Play Last Recording in Browser", font=("Segoe UI",9), width=32, command=play_in_browser, bg="#3b82f6", fg="white", activebackground="#1d4ed8", activeforeground="white", relief="flat", padx=5, pady=4)
    web_btn.pack(pady=(0,8))
    status_label = tk.Label(root, text="Idle", font=("Segoe UI",10), bg="#111827", fg="#22c55e")
    status_label.pack(pady=(0,6))
    root.bind("<Control-r>", start_recording)
    root.bind("<Control-s>", stop_recording)
    root.protocol("WM_DELETE_WINDOW", on_close)
    voice_thread = threading.Thread(target=voice_listener, daemon=True)
    voice_thread.start()
    root.mainloop()

def on_close():
    stop_recording()
    os._exit(0)

if __name__ == "__main__":
    launch_gui()

