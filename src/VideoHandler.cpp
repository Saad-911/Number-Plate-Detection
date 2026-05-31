#include"VideoHandler.h"
#include <iostream>

using namespace std;
using namespace cv;

VideoHandler::VideoHandler(const string& inputPath, const string& outputPath) {
    cap.open(inputPath);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Could not open video: " << inputPath << "\n";
        return;
    }

    // Extract video properties
    fps = (int)cap.get(CAP_PROP_FPS);
    width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    // Initialize the writer to save the output video safely
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(outputPath, codec, fps, Size(width, height));
}

VideoHandler::~VideoHandler() {
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
}

bool VideoHandler::readFrame(Mat& frame) {
    return cap.read(frame);
}

void VideoHandler::writeFrame(const Mat& frame) {
    if (writer.isOpened()) {
        writer.write(frame);
    }
}

bool VideoHandler::isOpened() const {
    return cap.isOpened() && writer.isOpened();
}

int VideoHandler::getTotalFrames() const {
    return totalFrames;
}
