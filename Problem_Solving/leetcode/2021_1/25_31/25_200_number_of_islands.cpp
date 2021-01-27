#include<iostream>
#include<vector>
#include<queue>
using namespace std;

const int dir[4][2] = {
    {-1,0},{1,0},{0,1},{0,-1}
};

class Node{
    public:
    int i,j;
    Node(int _i,int _j):i(_i),j(_j){}
};

inline bool in_bound(int i, int j, int m, int n){
    if(i>=0&&i<m)
        if(j>=0&&j<n)
            return true;
    return false;
}

class Solution {
public:
    int cnt;
    int m,n;
    queue<Node*> q;
    Node * front;
    int numIslands(vector<vector<char>>& grid) {
        cnt=0;
        m=grid.size();
        n=grid[0].size();
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                if(grid[i][j]==1)
                {
                    cnt++;
                    grid[i][j]=0;
                    q.push(new Node(i,j));
                    while(!q.empty())
                    {
                        front=q.front();
                        q.pop();
                        for(int a=0;a<4;a++)
                        {
                            int next_i =front->i+dir[a][0];
                            int next_j =front->j+dir[a][1];
                            if(in_bound(next_i,next_j,m,n))
                                if(grid[next_i][next_j])
                                {
                                    grid[next_i][next_j]=0;
                                    q.push(new Node(next_i,next_j));
                                }
                        }
                    }
                }
        return cnt;
    }
};

int main()
{
    freopen("200_input.txt","r",stdin);
    int m,n;
    cin>>m>>n;
    
}