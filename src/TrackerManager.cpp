#include "TrackerManager.h"

using namespace std;
using namespace cv;
using namespace dnn;

void TrackerManager::initializeTrackers(const Mat& frame, const vector<Rect>& detections, const vector<string>& texts) {
    activeTrackers.clear();

    // Create rectangle representing the exact size of the video frame
    Rect imageBounds(0, 0, frame.cols, frame.rows);

    for (size_t i = 0; i < detections.size(); ++i) {
        TrackedObject obj;
        obj.tracker = TrackerMIL::create();

        Rect clampedBox = detections[i] & imageBounds;

        if (clampedBox.area() <= 0) continue;

        obj.bbox = clampedBox;

        // If plate has text, save it; otherwise use a placeholder
        obj.text = (i < texts.size()) ? texts[i] : "DETECTING...";

        obj.tracker->init(frame, obj.bbox);
        activeTrackers.push_back(obj);
    }
}

void TrackerManager::updateTrackers(const Mat& frame) {
    for (auto& obj : activeTrackers) {
        obj.tracker->update(frame, obj.bbox);
    }
}

const vector<TrackedObject>& TrackerManager::getTrackedObjects() const {
    return activeTrackers;
}
