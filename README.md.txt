# Driver Drowsiness Detection

A real-time driver drowsiness monitoring system built using **Python**, **OpenCV**, **MediaPipe**, and a custom-trained **deep learning eye-state model**.  
The system detects eye closure, blinks, and signs of drowsiness, then plays an alert sound to wake the driver.

---

## 🚀 Features
- Real-time webcam-based eye detection  
- Deep learning model for eye state classification  
- Continuous drowsiness monitoring  
- Instant alarm using a `.wav` alert sound  
- Lightweight and runs smoothly on CPU  

---

## Limitations

Lighting Issues: Poor or uneven lighting reduces detection accuracy.

Hardware Load: Low-performance systems cause processing delays.

Face Coverage: Covered or angled faces reduce landmark accuracy.

Limited Signals: Only eyes, yawning, and head tilt used for drowsiness.

IP Camera Lag: Network delay causes dropped frames and slow alerts.
