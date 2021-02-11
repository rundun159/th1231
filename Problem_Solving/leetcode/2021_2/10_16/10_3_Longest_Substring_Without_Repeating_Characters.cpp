#include <iostream>
#include <vector>
#include <queue>
#define DEBUG false
using namespace std;

void show_bool_list(vector<bool> & make_list)
{
    for(int i=0;i < make_list.size(); i++)
        if(make_list[i])
            cout<<i<<" ";
    cout<<endl;
}

class Solution {
public:
    static int lengthOfLongestSubstring(string s) {
        int len = s.size();
        int start_idx, end_idx, ret, mask, now_max;
        if ( len==0 || len==1)
            return len;
        vector<char> input_s(len,0);
        vector<bool> mask_list(256,false);
        for(int i=0;i<len;i++)
            input_s[i] = s[i];

        if(DEBUG)
        {
            cout<<"input_s"<<endl;
            for(int i=0;i<len;i++)
                cout<<(int)input_s[i]<<" ";
            cout<<endl;
        }
        start_idx = 0, end_idx = -1, ret=1, mask=0, now_max=0;

        while(end_idx<(len-1))
        {
            end_idx++;
            if(DEBUG)
            {
                cout<<start_idx<<" "<<end_idx<<" "<<mask_list[input_s[end_idx]]<<" "<<now_max<<endl;
                show_bool_list(mask_list);
            }
            if(mask_list[input_s[end_idx]])
            {
                int next_start_idx = start_idx;
                while(input_s[next_start_idx] != input_s[end_idx])
                    next_start_idx++;
                for(int i = start_idx; i < next_start_idx; i++)
                    mask_list[input_s[i]] = false;
                start_idx = next_start_idx + 1;
                now_max = end_idx - start_idx + 1;
            }
            else
            {
                mask_list[input_s[end_idx]] = true;
                now_max++;
                ret = now_max>ret ? now_max : ret;
            }            
        }
        return ret;
    }
};

// class Solution {
// public:
//     static int lengthOfLongestSubstring(string s) {
//         int len = s.size();
//         int start_idx, end_idx, ret, mask, now_max;
//         if ( len==0 || len==1)
//             return len;
//         vector<int> input_s(len,0);
//         for(int i=0;i<len;i++)
//         {
//             input_s[i] = (s[i]-'a');
//             if(DEBUG)
//             {
//                 cout<<s[i];
//                 cout<<(s[i]-'a');
//             }
//         }
//         if(DEBUG)
//             for(int i=0;i<len;i++)
//                 cout<<input_s[i]<<" ";

//         start_idx = 0, end_idx = -1, ret=1, mask=0, now_max=0;

//         while(end_idx<(len-1))
//         {
//             end_idx++;
//             if(DEBUG)
//                 cout<<start_idx<<" "<<end_idx<<endl;
//             if(mask&(1<<input_s[end_idx]))
//             {
//                 int next_start_idx = start_idx;
//                 while(input_s[next_start_idx]!=input_s[end_idx])
//                     next_start_idx++;
//                 start_idx = next_start_idx+1;
//                 now_max = (end_idx - start_idx) + 1;
//                 mask=0;
//                 for(int i=start_idx;i<=end_idx;i++)
//                     mask |= 1<<input_s[i];
//             }
//             else
//             {
//                 mask |= 1<<input_s[end_idx];
//                 now_max++;
//                 ret = now_max>ret ? now_max : ret;
//             }                        
//         }
//         return ret;
//     }
// };

int main()
{
    freopen("10_3_Longest_Substring_Without_Repeating_Characters.txt","r",stdin);
    string s;
    cin>>s;
    cout<<Solution::lengthOfLongestSubstring(s)<<endl;
}