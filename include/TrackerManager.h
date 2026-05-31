#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

struct TrackedObject { //tracker
    Ptr<Tracker> tracker;
    Rect bbox;
    string text;
};

class TrackerManager {
private:
    vector<TrackedObject> activeTrackers;

public:
    void initializeTrackers(const Mat& frame, const vector<Rect>& detections, const vector<string>& texts);
    void updateTrackers(const Mat& frame);
    const vector<TrackedObject>& getTrackedObjects() const;
};
