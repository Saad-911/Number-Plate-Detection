#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace dnn;

class PlateDetector {
private:
    Net net;
    float confidenceThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;

public:
    PlateDetector(const string& modelPath, float confThreshold = 0.5f, float nmsThresh = 0.4f);
    vector<Rect> detect(Mat& frame);
};