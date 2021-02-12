#include <iostream>
#include <vector>
#include <queue>
#include <map>
#define DEBUG false
#define MAX_ID 1<<20
#define EXTENT 1<<10
using namespace std;

class UndergroundSystem {
public:
    int capacity;
    int station_cnt;
    vector<vector<int>> move_cnt;
    vector<vector<int>> total_time;
    vector<int> start_station;
    vector<int> start_time;    
    map<string,int> station_map;
    void extend_vector()
    {
        int next_capacity = capacity + (EXTENT);
        vector<int> new_vector(EXTENT,0);
        vector<vector<int>> new_vector2(EXTENT, vector<int>(next_capacity,0));
        for(int i=0;i<capacity;i++)
        {
            move_cnt[i].insert(move_cnt[i].end(),new_vector.begin(),new_vector.end());
            total_time[i].insert(total_time[i].end(),new_vector.begin(),new_vector.end());
        }
        move_cnt.insert(move_cnt.end(),new_vector2.begin(),new_vector2.end());
        total_time.insert(total_time.end(),new_vector2.begin(),new_vector2.end());       
        capacity += EXTENT;        
    }
    UndergroundSystem() {
        capacity = EXTENT;
        station_cnt = 0;
        move_cnt = vector<vector<int>>(capacity,vector<int>(capacity,0));
        total_time = vector<vector<int>>(capacity,vector<int>(capacity,0));
        start_station = vector<int>(MAX_ID,-1);
        start_time= vector<int>(MAX_ID,-1);
    }
    
    void checkIn(int id, string stationName, int t) {
        int now_station_idx;
        if(station_map.count(stationName)==0)
        {
            station_map[stationName] = station_cnt ++ ; 
            if(station_cnt < capacity)
                extend_vector();
        }
        now_station_idx = station_map[stationName];
        if(DEBUG)
            cout<<"now_station_idx : "<<now_station_idx<<" stationName : "<<stationName<<endl;
        start_station[id] = now_station_idx;
        start_time[id] = t;
    }
    
    void checkOut(int id, string stationName, int t) {
        if(station_map.count(stationName)==0)
        {
            station_map[stationName] = station_cnt ++ ; 
            if(station_cnt < capacity)
                extend_vector();
        }
        int move_time;
        int start_station_idx, end_station_idx;

        move_time = t - start_time[id];
        start_station_idx = start_station[id];
        end_station_idx = station_map[stationName];

        move_cnt[start_station_idx][end_station_idx]++;
        total_time[start_station_idx][end_station_idx] += move_time;
        if(DEBUG)
            cout<<"start_station_idx : "<<start_station_idx<<" end_station_idx : "<<end_station_idx<<" move_time : "<<move_time<<endl;
    }
    
    double getAverageTime(string startStation, string endStation) {
        int start_station_idx, end_station_idx;
        start_station_idx = station_map[startStation];
        end_station_idx = station_map[endStation];
        // cout<<"startStation : "<<startStation<<" start_station_idx : "<<start_station_idx<<" endStation : "<<endStation<<" end_station_idx : "<<end_station_idx
        if(DEBUG)
            cout<<"total_time : "<<total_time[start_station_idx][end_station_idx]<<" move_cnt :"<<move_cnt[start_station_idx][end_station_idx]<<endl;
        if(move_cnt[start_station_idx][end_station_idx]==0)
            return (double)0.0;
        else
            return (double)total_time[start_station_idx][end_station_idx]/move_cnt[start_station_idx][end_station_idx];
    }
};

/**
 * Your UndergroundSystem object will be instantiated and called as such:
 * UndergroundSystem* obj = new UndergroundSystem();
 * obj->checkIn(id,stationName,t);
 * obj->checkOut(id,stationName,t);
 * double param_3 = obj->getAverageTime(startStation,endStation);
 */