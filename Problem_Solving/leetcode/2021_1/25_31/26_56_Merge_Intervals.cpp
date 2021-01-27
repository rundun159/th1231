#include<iostream>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;

class Node
{
    public:
        int x,y;
        Node(int _x,int _y):x(_x),y(_y){}
};

class Intervals
{
    public:
        int start, end;
        Intervals(int _s, int _e):start(_s),end(_e){}
        bool merged(Node * new_node)
        {
            if (new_node->x<=end)
            {
                if(new_node->y>end)
                    end=new_node->y;
                return true;
            }
            else
                return false;
        }
};

bool compare_node(Node *&a, Node *&b)
{
    return a->x<b->x;
}

class Solution {
public:
    static vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int m = intervals.size();
        vector<Node *> node_list(m,NULL);
        vector<vector<int>> ret_list;

        for(int i=0;i<m;i++)
            node_list[i]= new Node(intervals[i][0],intervals[i][1]);
        sort(node_list.begin(),node_list.end(),compare_node);

        vector<Intervals *> intervals_list(1,new Intervals(node_list[0]->x,node_list[0]->y));
        delete(node_list[0]);
        Intervals * back =intervals_list.back();
        for(int i=1;i<m;i++)
        {
            if(!back->merged(node_list[i]))
            {
                intervals_list.push_back(new Intervals(node_list[i]->x,node_list[i]->y));
                back = intervals_list.back();
            }            
            delete(node_list[i]);
        }
        ret_list = vector<vector<int>>(intervals_list.size(),vector<int>(2,0));
        for(int i=0;i<intervals_list.size();i++)
        {
            ret_list[i][0] = intervals_list[i]->start;
            ret_list[i][1] = intervals_list[i]->end;
        }
        return ret_list;
    }
};

int main()
{
    freopen("26_56_Merge_Intervals_input.txt","r",stdin);
    int m,n;
    cin>>m;
    vector<vector<int>> intervals(m,vector<int>(2,0));
    for(int i=0;i<m;i++)
        cin>>intervals[i][0]>>intervals[i][1];
    Solution::merge(intervals);
}