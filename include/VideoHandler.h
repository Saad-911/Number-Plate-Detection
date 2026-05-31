#pragma once
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class VideoHandler {
private:
    VideoCapture cap;
    VideoWriter writer;
    int fps;
    int width;
    int height;
    int totalFrames;

public:
    
    VideoHandler(const string& inputPath, const string& outputPath);
    ~VideoHandler();
    
    bool readFrame(Mat& frame);
    void writeFrame(const Mat& frame);

    bool isOpened() const;
    int getTotalFrames() const;
};
