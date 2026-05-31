# High-Performance ALPR System (C++ & YOLOv8)

An optimized, real-time Automatic License Plate Recognition (ALPR) system built entirely in C++. This project leverages OpenCV's DNN module to run dual YOLOv8 models (Plate Detection + Character OCR) natively on the CPU, utilizing frame-skipping and MIL tracking to achieve maximum performance without requiring a dedicated GPU.

---

## Features
* **Dual YOLOv8 Architecture:** Utilizes two ONNX-exported YOLO models—one for localizing license plates and a second for reading individual alphanumeric characters.
* **Real-Time Speed Optimization:** Implements a frame-skipping algorithm paired with OpenCV's TrackerMIL. AI detection runs periodically, while lightweight mathematical tracking handles the intermediate frames.
* **Non-Maximum Suppression (NMS):** Built-in NMS filtering for both bounding boxes and character OCR to eliminate overlapping, duplicate predictions.
* **Automated Data Logging:** Seamlessly logs unique license plates alongside system timestamps to a .csv file for database integration.
* **Modular OOP Design:** Clean separation of concerns (I/O, Detection, Tracking, Logging) for highly maintainable and scalable code.

---

## Project Architecture

The codebase is structured into strict modules separating declarations (include/) from implementations (src/):

```text
├── include/                 # Header files (Class declarations)
│   ├── Logger.h             # Static CSV logging utility
│   ├── PlateDetector.h      # YOLOv8 plate detection pipeline
│   ├── PlateReader.h        # YOLOv8 character OCR pipeline + NMS
│   ├── TrackerManager.h     # MIL tracking and bounding box clamping
│   └── VideoHandler.h       # Safely wraps OpenCV VideoCapture/Writer
│
├── src/                     # Source files (Implementations)
│   ├── Logger.cpp
│   ├── PlateDetector.cpp
│   ├── PlateReader.cpp
│   ├── TrackerManager.cpp
│   ├── VideoHandler.cpp
│   └── main.cpp             # Orchestrates the main processing loop
│
├── .gitignore
├── LICENSE
└── README.md
