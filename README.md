````markdown
# ⭐ Smart Screen Recorder Advanced  
### A Powerful Python-Based Screen, Audio & Webcam Recorder  
**© 2025 Kumar Shivam — All Rights Reserved**

---

## ⭐ Overview  
Smart Screen Recorder Advanced is a Python-based desktop recording tool that captures:

- ⭐ Full screen  
- ⭐ Webcam overlay  
- ⭐ Microphone audio  
- ⭐ (Optional) System/loopback audio  
- ⭐ Real-time logs and advanced controls  

It provides a clean modern UI, FFmpeg-powered final encoding, adjustable screen speed, playback-speed control, and voice-activated commands.

---

## ⭐ Features  
### 🎥 Screen & Webcam  
- Records full display with optional webcam overlay  
- Adjustable overlay **size**, **position**, and **border**  
- Live webcam preview  

### 🎤 Audio  
- Optional **microphone recording**  
- Optional **system/loopback audio recording**  
- Device selection for both sources  

### ⚙ Recording Controls  
- Multiple output formats: **MP4**, **MKV**, **MOV**  
- Adjustable **screen speed** (10–60 FPS presets)  
- Optional **fixed-duration recording**  
- Playback speed modification (0.5x → 4x)  
- Real-time status, timer, and logging panel  

### 🗣 Voice & Hotkeys  
- Voice commands: **"start"**, **"stop"**  
- Hotkeys:  
  - **Ctrl + R** = Start  
  - **Ctrl + S** = Stop  

### 🧪 Processing  
- FFmpeg-powered merging & encoding  
- Cursor highlight  
- Faststart enabled for web playback  

---

## ⭐ Requirements  

- **Python 3.8+**
- **FFmpeg** (must be installed and available on PATH)
- Python Libraries:
  - pyautogui  
  - opencv-python  
  - sounddevice  
  - scipy  
  - ffmpeg-python  
  - pillow  
  - pyttsx3  
  - SpeechRecognition  
  - numpy  
  - tkinter (bundled with most Python installs)

Install dependencies:

```bash
pip install pyautogui opencv-python sounddevice scipy ffmpeg-python pillow pyttsx3 SpeechRecognition numpy
````

---

## ⭐ Installation

1. Install **Python 3.8+**
2. Install required packages (see above)
3. Install **FFmpeg**
4. Run:

```bash
python recorder.py
```

---

## ⭐ Usage

1. Launch:

```bash
python recorder.py
```

2. Configure settings:

   * Output format
   * Screen speed
   * Audio sources
   * Webcam options
   * Duration
   * Playback speed

3. Start recording:

   * Click **Start Recording**
   * OR press **Ctrl + R**
   * OR say **"start"**

4. Stop recording:

   * Click **Stop**
   * OR press **Ctrl + S**
   * OR say **"stop"**

5. Your recording will be exported to the chosen output folder.

---

## ⭐ Options & Settings

### **Screen Options**

* Screen speed presets:
  ⭐ 10 fps • ⭐ 20 fps • ⭐ 30 fps • ⭐ 45 fps • ⭐ 60 fps

### **Webcam Options**

* Enable/Disable webcam
* Position:
  ⭐ Top-left • ⭐ Top-right • ⭐ Bottom-left • ⭐ Bottom-right
* Size (% of screen width)
* Border toggle

### **Audio Options**

* Microphone capture
* System audio (loopback)
* Device selection

### **Advanced**

* Playback speed
* Suppress dialogs
* Real-time log viewer

---

## ⭐ Troubleshooting

### **FFmpeg not found**

Install FFmpeg and ensure running:

```bash
ffmpeg -version
```

### **Audio not recording**

* Refresh device list
* Check device index
* Some systems do not support loopback audio

### **Webcam errors**

* Try device index 0, 1, 2…
* Ensure no app is using the camera

### **High CPU usage**

* Lower screen speed
* Reduce webcam size
* Disable webcam altogether

---

## ⭐ Known Limitations

* System audio may not work on all OSes
* GUI uses Tkinter (appearance varies by OS)
* Voice recognition requires internet
* Heavy CPU usage at high FPS

---

## ⭐ Project Status

⚠️ **This tool is NOT fully optimized yet.**
Expect improvements over time.
Thank you for your patience and support!

---

## ⭐ License

```
© 2025 Kumar Shivam — All Rights Reserved
```

---

```
```
