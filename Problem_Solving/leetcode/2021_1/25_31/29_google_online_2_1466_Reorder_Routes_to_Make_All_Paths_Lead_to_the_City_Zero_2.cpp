#include <iostream>
#include <vector>
#include <queue>
using namespace std;
class Node
{
public:
	int in_e_num;
	int out_e_num;
	vector<int> in_e_list;
	vector<int> out_e_list;
	Node():in_e_num(0),out_e_num(0){}
	void add_in_e(int node) { in_e_num++; in_e_list.push_back(node); }
	void add_out_e(int node) { out_e_num++; out_e_list.push_back(node); }
};

class Solution {
public:
    static int minReorder(int n, vector<vector<int>>& connections) {
	    vector<Node *> node_list(n,new Node());
		vector<bool> visited(n,false);
		int ret=0;
		Node * now_node;
	    for(int i=0;i<n-1;i++)
	    {
		int x = connections[i][1], y = connections[i][0];
		node_list[x]->add_out_e(y);
		node_list[y]->add_in_e(x);
	    }
	    queue<int> q_r;
		queue<int> q_a;
		queue<int> q_b;
		
		visited[0] = true;
		q_r.push(0);
		while(!q_r.empty())
		{
			int now_node_idx = q_r.front();
			q_r.pop();
			now_node = node_list[now_node_idx];
			if(now_node->in_e_num!=0)
				for(int in_idx : now_node->in_e_list)
				{
					visited[in_idx] = true;
					q_a.push(in_idx);
				}
			if(now_node->out_e_num!=0)
				for(int out_idx : now_node->out_e_list)
				{
					visited[out_idx] = true;
					q_r.push(out_idx);
				}
		}
		int mode = 0 ;
		while(1)
		{
			if(mode==0)
			{
				// make full use of q_a
				while(!q_a.empty())
				{
					int now_a_idx = q_a.front();
					q_a.pop();
					
				}
			}
			else if(mode==1)
			{
				// make full use of q_b
				while(!q_b.empty())
				{
					
				}
			}
			mode ^= 1;
		}

    }
};

int main()
{
	freopen("29_google_online_2_1466_Reorder_Routes_to_Make_All_Paths_Lead_to_the_City_Zero_input.txt","r",stdin);
	int n;
	vector<vector<int>> connections(n-1,vector<int>(2,0));
	for(int i=0;i<n-1;i++)
		cin>>connections[i][0]>>connections[i][1];
	cout<<Solution::minReorder(n,connections)<<endl;	
	return 0;
}
