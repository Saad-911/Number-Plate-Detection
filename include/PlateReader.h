#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn;

class PlateReader {
private:
    Net net;
    float confidenceThreshold;
    int inputWidth;
    int inputHeight;

    vector<char> classNames;

public:
    PlateReader(const string& modelPath, float confThreshold = 0.5f);
    string readText(const Mat& plateCrop);
};
