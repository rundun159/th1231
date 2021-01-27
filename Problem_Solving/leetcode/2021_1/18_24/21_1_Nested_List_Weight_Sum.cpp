#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;


class NestedInteger {
    public:
    // Constructor initializes an empty nested list.
    NestedInteger();
    // Constructor initializes a single integer.
    NestedInteger(int value);
    // Return true if this NestedInteger holds a single integer, rather than a nested list.
    bool isInteger() const;
    // Return the single integer that this NestedInteger holds, if it holds a single integer
    // The result is undefined if this NestedInteger holds a nested list
    int getInteger() const;
    // Set this NestedInteger to hold a single integer.
    void setInteger(int value);
    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    void add(const NestedInteger &ni);
    // Return the nested list that this NestedInteger holds, if it holds a nested list
    // The result is undefined if this NestedInteger holds a single integer
    const vector<NestedInteger> &getList() const;
};

// class meta
// {
//     public:
//     int depth;
//     long long int score;
//     meta(int init_depth, int init_score){
//         depth = init_depth;
//         score = init_score;
//     }
// };

// void cal_score(NestedInteger const & n_int, meta prev_meta, meta & total_meta)
// {
//     if(n_int.isInteger())
//         total_meta.score += (prev_meta.depth+1) * n_int.getInteger();
//     else
//     {
//         const vector<NestedInteger> & inside_n_int_linst = n_int.getList();
//         for(NestedInteger const & inside_n_int : inside_n_int_linst)
//         {
//             meta temp_meta(prev_meta);
//             temp_meta.depth+=1;
//         }
//     }
    
// }

void cal_score(NestedInteger const & n_int, int prev_depth, int & total_score)
{
    if(n_int.isInteger())
        total_score += (prev_depth+1) * n_int.getInteger();
    else
    {
        const vector<NestedInteger> & inside_n_int_linst = n_int.getList();
        for(NestedInteger const & inside_n_int : inside_n_int_linst)
            cal_score(inside_n_int, prev_depth+1,total_score);
    }    
}

class Solution {
public:
    int depthSum(vector<NestedInteger>& nestedList) {
        int total_score=0;
        int now_depth=0;
        for(NestedInteger const & inside_n_int : nestedList)
            cal_score(inside_n_int, now_depth, total_score);
        return total_score;
    }
};