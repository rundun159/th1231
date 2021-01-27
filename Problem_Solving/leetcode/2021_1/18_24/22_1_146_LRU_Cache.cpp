#include <iostream>
#include <vector>
#include <stdio.h>
#include <queue>
#include <list>
#include <unordered_map>
using namespace std;
#define MAX_KEY 3000
#define EMPTY -1
#define DEBUG 0
#define DEBUG2 1
class Node{
public:
    int cnt, key;
    Node(int _cnt, int _key):cnt(_cnt),key(_key){}
};

class LRUCache {
public:
    int cnt;
    int capacity;
    int now_cap;
    queue<Node *> q;
    vector<int> last_update;
    vector<int> values;
    LRUCache(int _capacity) {
        cnt = 0;
        now_cap = 0;
        last_update = vector<int>(MAX_KEY+1,EMPTY);
        values = vector<int>(MAX_KEY+1,EMPTY);
        capacity = _capacity;
    }    
    int get(int key) {
        cnt++;
        if(DEBUG)
        {
            cout<<"in get function"<<endl;
            cout<<values[key]<<endl;
        }
        if(values[key]==EMPTY) 
            return -1;
        else{
            q.push(new Node(cnt,key));
            last_update[key]=cnt;
            return values[key];
        }
    }    
    void put(int key, int value) {
        if(DEBUG)
            cout<<"after if statement"<<endl;
        cnt++;
        if(now_cap==capacity)
        {
            if(DEBUG)
                cout<<now_cap<<", "<<capacity<<endl;
            bool done = false;
            int victim;
            Node * front;
            while(!done)
            {
                front = q.front();
                q.pop();
                if(front->cnt == last_update[front->key])
                {
                    done=true;
                    victim=front->key;
                }
                delete(front);
            }
            cout<<"Victim is "<<victim<<endl;
            last_update[victim]=EMPTY;
            values[victim]=EMPTY;
            now_cap--;
        }
        if(DEBUG)
            cout<<"after if statement"<<endl;
        q.push(new Node(cnt,key));
        last_update[key]=cnt;
        values[key]=value;
        now_cap++;
    }    
};

class LRUCache {
public:
    LRUCache(int capacity) : _capacity(capacity) {}
    
    int get(int key) {
        auto it = cache.find(key);
        if (it == cache.end()) return -1;
        touch(it);
        return it->second.first;
    }
    
    void set(int key, int value) {
        auto it = cache.find(key);
        if (it != cache.end()) touch(it);
        else {
			if (cache.size() == _capacity) {
				cache.erase(used.back());
				used.pop_back();
			}
            used.push_front(key);
        }
        cache[key] = { value, used.begin() };
    }
    
private:
    typedef list<int> LI;
    typedef pair<int, LI::iterator> PII;
    typedef unordered_map<int, PII> HIPII;
    
    void touch(HIPII::iterator it) {
        int key = it->first;
        used.erase(it->second.second);
        used.push_front(key);
        it->second.second = used.begin();
    }
    
    HIPII cache;
    LI used;
    int _capacity;
};

int main()
{
    int capacity = 2;
    int key = 1;
    int value = 1;
    if (DEBUG)
        cout<<"fine"<<endl;
    LRUCache lRUCache = LRUCache(2);
    lRUCache.put(1, 1); // cache is {1=1}
    lRUCache.put(2, 2); // cache is {1=1, 2=2}
    cout<<"get(1) "<<lRUCache.get(1)<<endl;    // return 1
    lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    cout<<"get(2) "<<lRUCache.get(2)<<endl;    // returns -1 (not found)
    lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    cout<<"get(1) "<<lRUCache.get(1)<<endl;    // return -1 (not found)
    cout<<"get(3) "<<lRUCache.get(3)<<endl;    // return 3
    cout<<"get(4) "<<lRUCache.get(4)<<endl;    // return 4
    return 0;
}