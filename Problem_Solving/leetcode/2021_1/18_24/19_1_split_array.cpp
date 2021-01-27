#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;
class Solution {
public:
    int sol_len;
    vector<int> sol_nums;
    vector<int> sums;
    vector<vector<int>> sol_cache;
    bool splitArray(vector<int>& nums);
    bool try_par(vector<int> const & par);
    bool nothing(vector<int> const & par)
    {
        int a=1;
    }
};
int main()
{
    freopen("19_1_split_array_input.txt","r",stdin);
    int num;
    cin>>num;
    Solution sol;
    vector<int> nums(num);
    for(int i=0;i<num;i++)
        cin>>nums[i];
    cout<<sol.splitArray(nums)<<endl;
}

bool Solution::splitArray(vector<int>& nums)
{
    int len = nums.size();
    sol_len = len;
    sol_nums = nums;
    if(len<5)
        return false;
    sol_cache = vector<vector<int>>(len,vector<int>(len,0));
    for(int i=0;i<len;i++)
        sol_cache[i][i]=sol_nums[i];
    for(int i=1;i<len;i++)
        sol_cache[0][i] = sol_cache[0][i-1] + sol_nums[i];
    for(int i=1;i<len;i++)
        for(int j=i+1;j<len;j++)
            sol_cache[i][j] = sol_cache[i-1][j] - sol_nums[i-1];
    vector<int> par(3);
    sums = vector<int>(4,0);
    int start = 1, end = len-2;
    for(int i=start;i<=end-4;i++)
    {
        par[0] = i;
        for(int j=i+2;j<=end-2;j++)
        {
            par[1] = j;
            for(int k=j+2;k<=end;k++)
            {
                par[2]=k;
                nothing(par);
                if(try_par(par))
                    return true;
            }
        }            
    }
    return false;
}

bool Solution::try_par(vector<int> const & par)
{
    sums[0] = sol_cache[0][par[0]-1];
    sums[1] = sol_cache[par[0]+1][par[1]-1];
    if(sums[0]!=sums[1])
        return false;        
    sums[2] = sol_cache[par[1]+1][par[2]-1];
    if(sums[1]!=sums[2])
        return false;
    sums[3] = sol_cache[par[2]+1][sol_len-1];
    if(sums[2]!=sums[3])
        return false;
    else
        return true;
}