#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#define DEBUG false
using namespace std;

// class UndergroundSystem {
//     unordered_map<int, pair<string, int>> arrive;
//     map<pair<string, string>, vector<int>> leave;
// public:
//     UndergroundSystem() {
        
//     }
    
//     void checkIn(int id, string stationName, int t) {
//         // a customer can only check in one at time
//         if(arrive.find(id) == arrive.end())
//             arrive[id] = make_pair(stationName, t);
//     }
    
//     void checkOut(int id, string stationName, int t) {
//         if(arrive.find(id) != arrive.end()){
//             leave[{arrive.at(id).first, stationName}].push_back(t - arrive.at(id).second);
            
//             // remove customer id, so that after they checkout, they may check in again
//             arrive.erase(id);
//         }
//     }
    
//     double getAverageTime(string startStation, string endStation) {
//         double sum = 0.0;
//         vector<int> values = leave.at({startStation, endStation});
//         for(int &i: values) sum += i;
        
//         return (1.0 * sum) / values.size();
//     }
// };

class TIME_CNT{
    public:
        int total_time;
        int cnt;
        TIME_CNT(int first_time):total_time(first_time),cnt(1){}
        void add_time(int new_time)
        {
            cnt++;
            total_time+=new_time;
        }
        double ret_avg()
        {
            if(cnt=0)
                return 0;
            else
                return (double)total_time/cnt;
        }
};

class UndergroundSystem {
public:
    unordered_map<int, pair<string,int>> check_in;
    unordered_map<int, pair<string,int>>::iterator check_in_it;
    map<pair<string,string>, TIME_CNT> check_out;
    map<pair<string,string>, TIME_CNT>::iterator check_out_it;
    
    UndergroundSystem() {
        
    }
    
    void checkIn(int id, string stationName, int t) {
        if(check_in.find(id) == check_in.end())
            check_in[id] = {stationName, t};
    }
    
    void checkOut(int id, string stationName, int t) {
           check_in_it = check_in.find(id);
        if(check_in_it != check_in.end())
        {
            pair<string,string> station_pair = {check_in_it->second.first, stationName};

            check_out_it = check_out.find(station_pair);
            



            if(check_out_it == check_out.end())
            {
                // check_out[station_pair] = TIME_CNT();

            }
            check_in.erase(id);
        }
    }
    
    double getAverageTime(string startStation, string endStation) {
    }
};