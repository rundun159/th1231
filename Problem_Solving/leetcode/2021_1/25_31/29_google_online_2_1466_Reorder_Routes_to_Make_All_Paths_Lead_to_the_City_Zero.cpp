#include<iostream>
#include<vector>
#include<queue>
using namespace std;
class QNode{
public:
    int depth;
    int now_node;
    QNode(int _d, int _n): depth(_d), now_node(_n) {}
};
class Node
{
public:
    vector<int> edge_list;
    int out_edge_num;
    int in_edge_num;
    Node():out_edge_num(0),in_edge_num(0){}
};
class Solution {
public:
    static int minReorder(int n, vector<vector<int>>& connections) {
            vector<Node> node_list(n,Node());
            vector<int> tree_table(n,0);
            vector<int> node_tree_table(n,-1);
            vector<bool> visited(n,false);
            queue<QNode *> dfs_q;
            int tree_depth;
            int tree_idx = 0;
            int dfs_cnt = 0;
            int now_depth;

            for(int i=0;i<n;i++)
            {
                int y = connections[i][0], x = connections[i][1];
                node_list[x].edge_list.push_back(y);
                node_list[x].out_edge_num++;
                node_list[y].in_edge_num++;
            }
            
            node_tree_table[0] = 0 ;
            visited[0]=true;
            dfs_q.push(new QNode(0,0));
            while(!dfs_q.empty())
            {
                QNode * front_qnode = dfs_q.front();
                dfs_q.pop();
                for(auto node_idx : node_list[front_qnode->now_node].edge_list)
                    if(!visited[node_idx])
                    {
                        node_tree_table[node_idx] = tree_idx;
                        visited[node_idx] = true;
                        dfs_q.push(new QNode(0,node_idx));
                    }
                delete(front_qnode);
            }
            
            for(int i = 1; i<n;i++)
                if(!visited[i])
                {
                    tree_idx++;
                    node_tree_table[i] = tree_idx ;
                    visited[i]=true;
                    dfs_q.push(new QNode(0,i));
                    while(!dfs_q.empty())
                    {
                        QNode * front_qnode = dfs_q.front();
                        Node & now_node = node_list[front_qnode->now_node];
                        dfs_q.pop();
                        now_depth = front_qnode -> depth;
                        if(now_node.out_edge_num!=0)
                        {
                            for(auto node_idx : now_node.edge_list)
                                if(!visited[node_idx])
                                {
                                    node_tree_table[node_idx] = tree_idx;
                                    visited[node_idx] = true;
                                    dfs_q.push(new QNode(now_depth+1,node_idx));
                                }
                        }
                        else
                        {
                            if(now_node.in_edge_num!=0)
                            {
                                
                            }
                        }
                        
                        delete(front_qnode);
                    }
                }

        return 1;
    }
};

int main()
{
    freopen("29_google_online_2_1466_Reorder_Routes_to_Make_All_Paths_Lead_to_the_City_Zero_input.txt","r",stdin);
    int n;
    vector<vector<int>> connections(n,vector<int>(2,0));
    for(int i=0;i<n;i++)
        cin>>connections[i][0]>>connections[i][1];
    cout<<Solution::minReorder(n,connections)<<endl;
    
}