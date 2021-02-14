#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#define DEBUG false
using namespace std;

class UndergroundSystem {
    unordered_map<int, pair<string, int>> arrive;
    map<pair<string, string>, vector<int>> leave;
public:
    UndergroundSystem() {
        
    }
    
    void checkIn(int id, string stationName, int t) {
        // a customer can only check in one at time
        if(arrive.find(id) == arrive.end())
            arrive[id] = make_pair(stationName, t);
    }
    
    void checkOut(int id, string stationName, int t) {
        if(arrive.find(id) != arrive.end()){
            leave[{arrive.at(id).first, stationName}].push_back(t - arrive.at(id).second);
            
            // remove customer id, so that after they checkout, they may check in again
            arrive.erase(id);
        }
    }
    
    double getAverageTime(string startStation, string endStation) {
        double sum = 0.0;
        vector<int> values = leave.at({startStation, endStation});
        for(int &i: values) sum += i;
        
        return (1.0 * sum) / values.size();
    }
};