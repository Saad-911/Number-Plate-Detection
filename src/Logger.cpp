#include "Logger.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace std;

void Logger::saveToCSV(const string& filename, const vector<string>& plates) {
    if (plates.empty()) return;

    ofstream file;
    file.open(filename, ios::out | ios::app);

    if (!file.is_open()) {
        cerr << "[ERROR] Could not open log file: " << filename << "\n";
        return;
    }

    // Get the current system time for the timestamp
    auto now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
    localtime_s(&timeinfo, &now_c);

    for (const auto& plate : plates) {
        file << put_time(&timeinfo, "%Y-%m-%d %H:%M:%S") << "," << plate << "\n";
    }

    file.close();
    cout << "[SUCCESS] Logged " << plates.size() << " plates to CSV.\n";
}
