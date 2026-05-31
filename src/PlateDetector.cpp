#include "PlateDetector.h"
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

PlateDetector::PlateDetector(const string& modelPath, float confThreshold, float nmsThresh)
    : confidenceThreshold(confThreshold), nmsThreshold(nmsThresh), inputWidth(640), inputHeight(640) {

    cout << "Detecting Plate\n";
    net = readNetFromONNX(modelPath);

    // set to cpu (Change to GPU)
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
}

vector<Rect> PlateDetector::detect(Mat& frame) {
    Mat blob;
    // Resize to 640x640, scale pixels by 1/255, swap Red and Blue channels
    blobFromImage(frame, blob, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    Mat output0 = outputs[0];
    Mat output_matrix(output0.size[1], output0.size[2], CV_32F, output0.ptr<float>());
    Mat transposed;
    transpose(output_matrix, transposed);

    vector<Rect> boxes;
    vector<float> confidences;

    float x_factor = frame.cols / (float)inputWidth;
    float y_factor = frame.rows / (float)inputHeight;
    float* data = (float*)transposed.data;

    for (int i = 0; i < transposed.rows; ++i) {
        float confidence = data[4]; // index 4 is the confidence for a 1-class model
        if (confidence >= confidenceThreshold) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back(confidence);
        }
        data += transposed.cols;
    }

    // non maximum suppression (NMS) to remove overlapping boxes
    vector<int> nms_indices;
    NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, nms_indices);

    vector<Rect> final_boxes;
    for (int idx : nms_indices) {
        final_boxes.push_back(boxes[idx]);
    }

    return final_boxes;
}
