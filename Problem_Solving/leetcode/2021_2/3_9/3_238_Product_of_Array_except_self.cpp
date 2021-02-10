#include <iostream>
#include <vector>
#include <queue>
#define DEBUG false
using namespace std;

class Solution {
public:
    static vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> left_product(n,1), right_product(n,1), ret(n,1);
        for(int i=1;i<n;i++)
            left_product[i] = left_product[i-1] * nums[i-1];
        for(int i=n-2;i>=0;i--)
            right_product[i] = right_product[i+1] * nums[i+1];
        for(int i=0;i<n;i++)
            ret[i] = left_product[i] * right_product[i];
        return ret;
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