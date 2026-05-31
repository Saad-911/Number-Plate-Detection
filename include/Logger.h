#pragma once
#include <string>
#include <vector>

using namespace std;

class Logger {
public:
    static void saveToCSV(const string& filename, const vector<string>& plates);
};
