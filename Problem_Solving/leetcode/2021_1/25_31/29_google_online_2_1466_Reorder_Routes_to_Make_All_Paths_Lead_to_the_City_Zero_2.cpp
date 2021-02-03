#include <iostream>
#include <vector>
#include <queue>
#define DEBUG false
using namespace std;
class Node
{
public:
	int in_e_num;
	int out_e_num;
	int node_id;
	vector<int> in_e_list;
	vector<int> out_e_list;
	Node(int node_idx):node_id(node_idx),in_e_num(0),out_e_num(0){}
	void add_in_e(int node) { in_e_num++; in_e_list.push_back(node); }
	void add_out_e(int node) { out_e_num++; out_e_list.push_back(node); }
	void print_node_inf() {

		cout<<"node_id :"<<node_id<<" in_e_num : "<<in_e_num<<" out_e_num :"<<out_e_num<<endl;
		cout<<"print in_e_list"<<endl;
		for(int idx : in_e_list)
			cout<<idx<<endl;
		cout<<"print out_e_list"<<endl;		
		for(int idx : out_e_list)
			cout<<idx<<endl;
	}
};

class Solution {
public:
    static int minReorder(int n, vector<vector<int>>& connections) {
	    vector<Node *> node_list(n,nullptr);
		for(int i=0;i<n;i++)
			node_list[i]=new Node(i);
		vector<bool> visited(n,false);
		int ret=0;
		Node * now_node;
	    for(int i=0;i<n-1;i++)
	    {
			int x = connections[i][1], y = connections[i][0];
			node_list[x]->add_out_e(y);
			node_list[y]->add_in_e(x);
	    }
		if(DEBUG)
			for(auto node : node_list)
				node->print_node_inf();
	    queue<int> q_r;
		queue<int> q_a;
		queue<int> q_b;
	
		visited[0] = true;
		q_r.push(0);
		while(!q_r.empty())
		{
			int now_node_idx = q_r.front();
			// cout<<"q_r now_node_idx : "<<now_node_idx<<endl;
			q_r.pop();
			now_node = node_list[now_node_idx];
			if(now_node->in_e_num!=0)
				for(int in_idx : now_node->in_e_list)
					if(!visited[in_idx])
					{
						visited[in_idx] = true;
						q_a.push(in_idx);
					}
			if(now_node->out_e_num!=0)
				for(int out_idx : now_node->out_e_list)
					if(!visited[out_idx])
					{
						visited[out_idx] = true;
						q_r.push(out_idx);
					}
			}
		int mode = 0 ;
		bool finished = false;
		bool cnt_ret = false;
		while(!finished)
		{
			queue<int> * pop_queue;
			queue<int> * no_pop_queue;
			if(mode == 0)
			{
				pop_queue = &q_a;
				cnt_ret = true;
			}
			else if(mode == 1)
			{
				pop_queue = &q_b;
				cnt_ret = false;
			}
			if(pop_queue->empty())
				finished=true;
			else
			{
				while(!pop_queue->empty())
				{
					int now_idx = pop_queue->front();
					// cout<<"now_idx : "<<now_idx<<endl;
					pop_queue->pop();					
					if(cnt_ret)
					{
						ret++;
					}
					now_node = node_list[now_idx];
					for(int out_idx : now_node->out_e_list)
						//fill q_b
						if(!visited[out_idx])
						{
							visited[out_idx] = true;
							q_b.push(out_idx);
						}
					for(int in_idx : now_node->in_e_list)
						if(!visited[in_idx])
						{
							visited[in_idx] = true;
							q_a.push(in_idx);
						}
				}
			}
			mode ^= 1;
		}
		return ret;
    }
};

int main()
{
	freopen("29_google_online_2_1466_Reorder_Routes_to_Make_All_Paths_Lead_to_the_City_Zero_input.txt","r",stdin);
	int n;
	cin>>n;
	vector<vector<int>> connections(n-1,vector<int>(2,0));
	for(int i=0;i<n-1;i++)
		cin>>connections[i][0]>>connections[i][1];
	cout<<Solution::minReorder(n,connections)<<endl;	
	return 0;
}


			// if(mode==0)
			// {
			// 	// make full use of q_a
			// 	if(q_a.empty())
			// 		finished = true;
			// 	else
			// 		while(!q_a.empty())
			// 		{
			// 			int now_a_idx = q_a.front();
			// 			q_a.pop();
			// 			ret++;
			// 			now_node = node_list[now_a_idx];
			// 			for(int out_idx : now_node->out_e_list)
			// 				//fill q_b
			// 				if(!visited[out_idx])
			// 				{
			// 					visited[out_idx] = true;
			// 					q_b.push(out_idx);
			// 				}
			// 			for(int in_idx : now_node->in_e_list)
			// 				if(!visited[in_idx])
			// 				{
			// 					visited[in_idx] = true;
			// 					q_a.push(in_idx);
			// 				}
			// 		}
			// }
			// else if(mode==1)
			// {
			// 	// make full use of q_b
			// 	if(q_b.empty())
			// 		finished = true;
			// 	else
			// 		while(!q_b.empty())
			// 		{
			// 			int now_b_idx = q_b.front();
			// 			q_b.pop();
			// 			now_node = node_list[now_b_idx];
			// 			for(int out_idx : now_node->out_e_list)
			// 				//fill q_b
			// 				if(!visited[out_idx])
			// 				{
			// 					visited[out_idx] = true;
			// 					q_b.push(out_idx);
			// 				}
			// 			for(int in_idx : now_node->in_e_list)
			// 				if(!visited[in_idx])
			// 				{
			// 					visited[in_idx] = true;
			// 					q_a.push(in_idx);
			// 				}
						
			// 		}
			// }
