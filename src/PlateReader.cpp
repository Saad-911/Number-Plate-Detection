#include "PlateReader.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace dnn;
using namespace cv;

// A helper struct to sort characters left-to-right
struct DetectedChar {
    int x_center;
    char character;
    float confidence;
};

PlateReader::PlateReader(const string& modelPath, float confThreshold)
    : confidenceThreshold(confThreshold), inputWidth(640), inputHeight(640) {

    cout << "Detecting Characters\n";
    net = readNetFromONNX(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (char c : chars) {
        classNames.push_back(c);
    }
}

string PlateReader::readText(const Mat& plateCrop) {
    if (plateCrop.empty()) return "DETECTING...";

    Mat blob;
    blobFromImage(plateCrop, blob, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    Mat output0 = outputs[0];
    Mat output_matrix(output0.size[1], output0.size[2], CV_32F, output0.ptr<float>());
    Mat transposed;
    transpose(output_matrix, transposed);

    // vectors to hold raw predictions before NMS filtering
    vector<cv::Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    float* data = (float*)transposed.data;

    for (int i = 0; i < transposed.rows; ++i) {
        float maxClassScore = 0;
        int classId = -1;

        // loop through all class scores to find the highest one
        for (int j = 4; j < transposed.cols; ++j) {
            if (data[j] > maxClassScore) {
                maxClassScore = data[j];
                classId = j - 4;
            }
        }

        // If confident prediction, save the box
        if (maxClassScore >= confidenceThreshold && classId >= 0 && classId < classNames.size()) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = int(cx - 0.5 * w);
            int top = int(cy - 0.5 * h);

            boxes.push_back(cv::Rect(left, top, int(w), int(h)));
            confidences.push_back(maxClassScore);
            classIds.push_back(classId);
        }
        data += transposed.cols;
    }

    vector<int> nmsIndices;
    float nmsThreshold = 0.4f; // If boxes overlap by more than 40%, delete the weaker one
    NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, nmsIndices);

    vector<DetectedChar> finalChars;
    for (int idx : nmsIndices) {
        DetectedChar dc;
        dc.x_center = boxes[idx].x + (boxes[idx].width / 2);
        dc.character = classNames[classIds[idx]];
        dc.confidence = confidences[idx];
        finalChars.push_back(dc);
    }

    // Sort characters from left to right!
    sort(finalChars.begin(), finalChars.end(), [](const DetectedChar& a, const DetectedChar& b) {
        return a.x_center < b.x_center;
        });

    // Stitch the string together
    string finalPlate = "";
    for (const auto& dc : finalChars) {
        finalPlate += dc.character;
    }

    if (finalPlate.empty() || finalPlate.length() < 3) return "DETECTING...";
    return finalPlate;
}

