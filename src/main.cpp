#include <iostream>
#include <vector>
#include <set>
#include "VideoHandler.h"
#include "PlateDetector.h"
#include "PlateReader.h"
#include "TrackerManager.h"
#include "Logger.h"

using namespace std;
using namespace dnn;
using namespace cv;

int main() {
    // Update these paths
    string videoPath = "Input video path";      
    string outputVideo = "Output video path"; 
    string csvPath = "output csv path";   
    string plateModel = "plate detection onnx path";  
    string ocrModel = "Ocr path onnx";     

    int frameSkip = 5;
    int frameCount = 0;

    VideoHandler video(videoPath, outputVideo);
    if (!video.isOpened()) return -1;

    PlateDetector detector(plateModel, 0.5f, 0.4f);
    PlateReader ocr(ocrModel, 0.4f);
    TrackerManager trackerManager;

    Mat frame;
    vector<string> uniquePlatesList;
    set<string> seenPlates;

    cout << "Starting Loop\n";

    while (video.readFrame(frame)) {
        frameCount++;

        if (frameCount % 10 == 0) {
            cout << "\rProcessing Frame: " << frameCount << "/" << video.getTotalFrames() << flush;
        }

		// plate detection
        if (frameCount % frameSkip == 0) {
            vector<Rect> rawBoxes = detector.detect(frame);
            vector<string> currentTexts;

            for (const auto& box : rawBoxes) {
                // Crop and add padding
                int pad = 5;
                int x = max(0, box.x - pad);
                int y = max(0, box.y - pad);
                int w = min(frame.cols - x, box.width + (pad * 2));
                int h = min(frame.rows - y, box.height + (pad * 2));

                Rect tightBox(x, y, w, h);
                Mat crop = frame(tightBox);

                // Read the text
                string text = ocr.readText(crop);
                currentTexts.push_back(text);

                // Add to unique list new
                if (text != "DETECTING..." && seenPlates.find(text) == seenPlates.end()) {
                    seenPlates.insert(text);
                    uniquePlatesList.push_back(text);
                }
            }

            // Hand the boxes to the Tracker Manager
            trackerManager.initializeTrackers(frame, rawBoxes, currentTexts);
        }
        // tracking due to skipping frames
        else {
            trackerManager.updateTrackers(frame);
        }

        for (const auto& obj : trackerManager.getTrackedObjects()) {
            Rect box = obj.bbox;
            string text = obj.text;

            rectangle(frame, box, Scalar(0, 255, 0), 2);

            int baseLine;
            Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
            rectangle(frame, Point(box.x, max(0, box.y - textSize.height - 10)),
                Point(box.x + textSize.width, box.y), Scalar(0, 255, 0), FILLED);
            putText(frame, text, Point(box.x, max(0, box.y - 5)),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
        }

        video.writeFrame(frame);
    }

    cout << "\n\n[SUCCESS] Video processing complete!\n";

    Logger::saveToCSV(csvPath, uniquePlatesList);

    return 0;
}
