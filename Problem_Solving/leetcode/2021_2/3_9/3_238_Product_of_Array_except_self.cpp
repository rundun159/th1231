#include <iostream>
#include <vector>
#include <queue>
#define DEBUG false
using namespace std;

class Solution {
public:
    static vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        
    }
};
int main()
{
    freopen("3_238_Product_of_Array_except_self_input.txt","r",stdin);
    int n;
    vector<int> num(4);
    vector<int> output;
    for(int i=0;i<4;i++)
        cin>>num[i];
    output=Solution::productExceptSelf(num);
    for(auto o : output)
        cout<<o<<endl;
}