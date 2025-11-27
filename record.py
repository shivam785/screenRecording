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

# Globals
recording = False
voice_thread = None
profile_choice = "normal"  # "normal" or "web"
last_output_file = None

# Audio options
record_mic = True
record_system = False

# Output format option
output_extension = ".mp4"

# GUI widgets
root = None
start_btn = None
stop_btn = None
status_label = None
format_dropdown = None
ext_dropdown = None
mic_var = None
system_var = None

# Text-to-Speech
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def open_file(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", file_path])
        else:
            subprocess.call(["xdg-open", file_path])
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")


# -------- Voice Listener --------
def voice_listener():
    global recording
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        print("[WARN] Microphone not found — voice control disabled.")
        return

    while True:
        with mic as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3)
                command = recognizer.recognize_google(audio).lower()
                print(f"[Voice] Heard: {command}")

                if "start" in command and not recording:
                    speak("Okay, recording started")
                    start_recording()
                elif "stop" in command and recording:
                    speak("Recording stopped")
                    stop_recording()
                    time.sleep(1.5)
            except Exception:
                continue


# -------- Audio Recorder --------
def record_audio(audio_filename, use_mic: bool, use_system: bool):
    """
    Records audio from the default input device.
    `use_mic` / `use_system` are flags that could be used to route
    to different devices in a more advanced version.
    """
    global recording
    fs = 44100
    channels = 2
    frames = []

    if not use_mic and not use_system:
        print("[INFO] Audio recording disabled by user.")
        return

    print("[INFO] Audio recording started...")

    def callback(indata, frame_count, time_info, status):
        if status:
            print("[AUDIO STATUS]", status)
        frames.append(indata.copy())
        if not recording:
            raise sd.CallbackStop

    try:
        # NOTE:
        # For real system-audio capture, you'd typically select a loopback
        # device (e.g. Speakers (loopback)) via device=... in InputStream.
        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            while recording:
                sd.sleep(100)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print(f"[ERROR] Audio recording error: {e}")

    if frames:
        audio = np.concatenate(frames, axis=0)
        write(audio_filename, fs, audio)
        print("[INFO] Audio saved.")
    else:
        print("[WARN] No audio frames captured; skipping audio save.")


# -------- HTML video player (optional) --------
def create_and_open_player(video_path):
    video_path = Path(video_path).absolute()
    if not video_path.exists():
        messagebox.showerror("Error", "Video file does not exist.")
        return

    ext = video_path.suffix.lower().lstrip(".")
    mime_map = {
        "mp4": "video/mp4",
        "mov": "video/mp4",
        "mkv": "video/mp4",
        "avi": "video/x-msvideo",
    }
    mime_type = mime_map.get(ext, "video/mp4")

    html_path = video_path.with_name(video_path.stem + "_player.html")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Recording Player</title>
    <style>
        body {{
            background: #111827;
            color: #e5e7eb;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        h1 {{
            margin-bottom: 20px;
            font-size: 1.5rem;
        }}
        video {{
            max-width: 90%;
            max-height: 70vh;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        }}
    </style>
</head>
<body>
    <h1>🎥 Your Recording</h1>
    <video controls autoplay>
        <source src="{video_path.name}" type="{mime_type}">
        Your browser does not support the video tag.
    </video>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    webbrowser.open(html_path.as_uri())


# -------- Screen + Webcam Recorder --------
def record_screen(video_filename, audio_filename, final_output, profile, use_mic, use_system):
    """
    Records the screen + webcam, and (optionally) audio in parallel.
    After stopping, audio and video are merged automatically into `final_output`.
    User never has to manually combine.
    """
    global recording, last_output_file

    # Match screen resolution
    screen_width, screen_height = pyautogui.size()
    resolution = (screen_width, screen_height)

    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, resolution)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start_time = time.time()

    # Start audio thread only if at least one audio source is enabled
    audio_thread = None
    if use_mic or use_system:
        audio_thread = threading.Thread(
            target=record_audio,
            args=(audio_filename, use_mic, use_system),
        )
        audio_thread.start()
    else:
        print("[INFO] Audio disabled. Recording silent video only.")

    print("[INFO] Screen recording started...")
    update_status("Recording...", "red")
    update_buttons(True)

    try:
        while recording:
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            cursor_x, cursor_y = pyautogui.position()
            cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 255, 255), -1)

            ret, webcam_frame = cam.read()
            if ret:
                webcam_small = cv2.resize(webcam_frame, (320, 240))
                frame[10:10 + 240, 10:10 + 320] = webcam_small

            elapsed = int(time.time() - start_time)
            cv2.putText(
                frame,
                f"Elapsed: {elapsed}s",
                (10, resolution[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            out.write(frame)
            time.sleep(1 / fps)

    finally:
        out.release()
        cam.release()
        try:
            sd.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print("[INFO] Finalizing recording...")

        try:
            if use_mic or use_system and os.path.exists(audio_filename):
                # Merge audio + video into final output (single file).
                input_video = ffmpeg.input(video_filename)
                input_audio = ffmpeg.input(audio_filename)

                if profile == "web":
                    (
                        input_video
                        .filter('scale', '1280:-2')
                        .output(
                            input_audio,
                            final_output,
                            vcodec='libx264',
                            acodec='aac',
                            crf=26,
                            preset='veryfast',
                            movflags='+faststart',
                            strict='experimental'
                        )
                        .run(overwrite_output=True)
                    )
                else:
                    (
                        input_video
                        .output(
                            input_audio,
                            final_output,
                            vcodec='libx264',
                            acodec='aac',
                            crf=20,
                            preset='faster',
                            movflags='+faststart',
                            strict='experimental'
                        )
                        .run(overwrite_output=True)
                    )
            else:
                # No audio: just transcode/rename video into final container
                input_video = ffmpeg.input(video_filename)
                (
                    input_video
                    .output(
                        final_output,
                        vcodec='libx264',
                        crf=20 if profile == "normal" else 26,
                        preset='faster' if profile == "normal" else 'veryfast',
                        movflags='+faststart'
                    )
                    .run(overwrite_output=True)
                )

        except Exception as e:
            print(f"[ERROR] Error during finalization: {e}")
            messagebox.showerror("Error", f"Failed to finalize recording:\n{e}")
            update_status("Idle", "green")
            update_buttons(False)
            return
        finally:
            if os.path.exists(video_filename):
                os.remove(video_filename)
            if os.path.exists(audio_filename):
                os.remove(audio_filename)

        print(f"[✅] Final saved: {final_output}")
        last_output_file = os.path.abspath(final_output)

        messagebox.showinfo("Done", f"Recording saved as:\n{final_output}")
        update_status("Idle", "green")
        update_buttons(False)

        # Optional: open HTML player in browser
        try:
            create_and_open_player(last_output_file)
        except Exception as e:
            print(f"[WARN] Could not open browser player: {e}")


# -------- GUI helpers --------
def update_status(text, color="white"):
    global status_label
    if status_label and status_label.winfo_exists():
        status_label.after(0, lambda: status_label.config(text=text, fg=color))


def update_buttons(is_recording: bool):
    global start_btn, stop_btn
    if start_btn and stop_btn:
        def _update():
            if is_recording:
                start_btn.config(state="disabled")
                stop_btn.config(state="normal")
            else:
                start_btn.config(state="normal")
                stop_btn.config(state="disabled")
        start_btn.after(0, _update)


# -------- Start / Stop controls --------
def start_recording():
    global recording, profile_choice, record_mic, record_system, output_extension
    if recording:
        return
    if not format_dropdown or not ext_dropdown:
        return

    # Quality profile
    label = format_dropdown.get().lower()
    if "web" in label:
        profile_choice = "web"
    else:
        profile_choice = "normal"

    # Audio options from checkboxes
    record_mic = bool(mic_var.get())
    record_system = bool(system_var.get())

    # Output extension (from combobox)
    ext_label = ext_dropdown.get().lower()
    if "mkv" in ext_label:
        output_extension = ".mkv"
    elif "mov" in ext_label:
        output_extension = ".mov"
    else:
        output_extension = ".mp4"

    recording = True
    update_status("Preparing...", "orange")
    update_buttons(True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # temp video in mp4 container
    video_filename = f"temp_video_{timestamp}.mp4"
    audio_filename = f"temp_audio_{timestamp}.wav"
    final_output = f"Recording_{timestamp}{output_extension}"

    threading.Thread(
        target=record_screen,
        args=(video_filename, audio_filename, final_output, profile_choice, record_mic, record_system),
        daemon=True
    ).start()


def stop_recording():
    global recording
    if not recording:
        return
    print("[INFO] Stop requested...")
    recording = False
    update_status("Stopping...", "orange")
    update_buttons(False)


def play_in_browser():
    global last_output_file
    if last_output_file and os.path.exists(last_output_file):
        try:
            create_and_open_player(last_output_file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open in browser:\n{e}")
    else:
        messagebox.showwarning("No Recording", "No recording found yet. Please record first.")


# -------- GUI --------
def launch_gui():
    global root, start_btn, stop_btn, status_label, format_dropdown, ext_dropdown
    global mic_var, system_var

    root = tk.Tk()
    root.title("Smart Screen Recorder")
    root.resizable(False, False)

    # Window size & position (top center)
    window_width = 400
    window_height = 320
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    x = int((screen_width - window_width) / 2)
    y = 10
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    root.attributes("-topmost", True)
    root.configure(bg="#111827")

    # Title
    title_label = tk.Label(
        root,
        text="🎥 Smart Screen Recorder",
        font=("Segoe UI", 14, "bold"),
        bg="#111827",
        fg="white"
    )
    title_label.pack(pady=(10, 4))

    subtitle = tk.Label(
        root,
        text="Say 'start' / 'stop' or use the buttons below",
        font=("Segoe UI", 9),
        bg="#111827",
        fg="#9ca3af"
    )
    subtitle.pack(pady=(0, 10))

    # Options frame
    options_frame = tk.LabelFrame(
        root,
        text=" Options ",
        bg="#111827",
        fg="#e5e7eb",
        font=("Segoe UI", 9, "bold"),
        bd=1,
        relief="groove",
        padx=10,
        pady=8
    )
    options_frame.pack(padx=10, pady=5, fill="x")

    # Quality mode
    fmt_label = tk.Label(
        options_frame,
        text="Quality profile:",
        bg="#111827",
        fg="white",
        font=("Segoe UI", 9)
    )
    fmt_label.grid(row=0, column=0, padx=(0, 5), pady=2, sticky="w")

    format_dropdown = ttk.Combobox(
        options_frame,
        values=[
            "Normal MP4 (higher quality)",
            "Web MP4 (for websites, smaller size)"
        ],
        state="readonly",
        width=32
    )
    format_dropdown.current(1)
    format_dropdown.grid(row=0, column=1, pady=2, sticky="w")

    # Output extension
    ext_label = tk.Label(
        options_frame,
        text="Output format:",
        bg="#111827",
        fg="white",
        font=("Segoe UI", 9)
    )
    ext_label.grid(row=1, column=0, padx=(0, 5), pady=2, sticky="w")

    ext_dropdown = ttk.Combobox(
        options_frame,
        values=[
            "MP4 (.mp4, recommended)",
            "MKV (.mkv)",
            "MOV (.mov)"
        ],
        state="readonly",
        width=32
    )
    ext_dropdown.current(0)
    ext_dropdown.grid(row=1, column=1, pady=2, sticky="w")

    # Audio options
    audio_frame = tk.LabelFrame(
        root,
        text=" Audio Sources ",
        bg="#111827",
        fg="#e5e7eb",
        font=("Segoe UI", 9, "bold"),
        bd=1,
        relief="groove",
        padx=10,
        pady=8
    )
    audio_frame.pack(padx=10, pady=5, fill="x")

    mic_var = tk.BooleanVar(value=True)
    system_var = tk.BooleanVar(value=False)

    mic_chk = tk.Checkbutton(
        audio_frame,
        text="Record Microphone",
        variable=mic_var,
        bg="#111827",
        fg="white",
        activebackground="#111827",
        activeforeground="white",
        selectcolor="#1f2937",
        font=("Segoe UI", 9)
    )
    mic_chk.grid(row=0, column=0, sticky="w")

    system_chk = tk.Checkbutton(
        audio_frame,
        text="Record System Sound (requires loopback device)",
        variable=system_var,
        bg="#111827",
        fg="white",
        activebackground="#111827",
        activeforeground="white",
        selectcolor="#1f2937",
        font=("Segoe UI", 9),
        wraplength=230,
        justify="left"
    )
    system_chk.grid(row=1, column=0, sticky="w", pady=(2, 0))

    # Buttons
    btn_frame = tk.Frame(root, bg="#111827")
    btn_frame.pack(pady=12)

    start_btn = tk.Button(
        btn_frame,
        text="Start Recording",
        font=("Segoe UI", 10, "bold"),
        width=16,
        command=start_recording,
        bg="#22c55e",
        fg="white",
        activebackground="#16a34a",
        activeforeground="white",
        relief="flat",
        padx=5,
        pady=5
    )
    start_btn.grid(row=0, column=0, padx=6)

    stop_btn = tk.Button(
        btn_frame,
        text="Stop",
        font=("Segoe UI", 10, "bold"),
        width=10,
        command=stop_recording,
        bg="#ef4444",
        fg="white",
        activebackground="#b91c1c",
        activeforeground="white",
        relief="flat",
        padx=5,
        pady=5,
        state="disabled"
    )
    stop_btn.grid(row=0, column=1, padx=6)

    web_btn = tk.Button(
        root,
        text="▶ Play Last Recording in Browser",
        font=("Segoe UI", 9),
        width=32,
        command=play_in_browser,
        bg="#3b82f6",
        fg="white",
        activebackground="#1d4ed8",
        activeforeground="white",
        relief="flat",
        padx=5,
        pady=4
    )
    web_btn.pack(pady=(0, 8))

    status_label = tk.Label(
        root,
        text="Idle",
        font=("Segoe UI", 10),
        bg="#111827",
        fg="#22c55e"
    )
    status_label.pack(pady=(0, 6))

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def on_close():
    stop_recording()
    os._exit(0)


# -------- Entry Point --------
if __name__ == "__main__":
    voice_thread = threading.Thread(target=voice_listener, daemon=True)
    voice_thread.start()
    launch_gui()
