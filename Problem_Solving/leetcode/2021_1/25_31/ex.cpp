#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stack>
using namespace std;
typedef unsigned long int ulint;

void print_2n(uint a)
{
    stack<int> p_a;
    while(a!=0)
    {
        if(a%2==1)
            p_a.push(1);
        else
            p_a.push(0);
        a/=2;
    }    
    while(!p_a.empty())
    {
        cout<<p_a.top();
        p_a.pop();
    }
    cout<<endl;
}

void *ut_align(const void *ptr, /*!< in: pointer */
               ulint align_no)  /*!< in: align by this number */
{
    uint left =(((ulint)ptr)+1 + align_no - 1);
    uint right =  ~(align_no - 1);
    cout<<left<<" , ";
    print_2n(left);
    cout<<right<<" , ";
    print_2n(right);
    cout<<(left&right)<<" , ";
    print_2n(left&right);
    cout<<(ulint)(ptr)<<" , ";
    print_2n((ulint)(ptr));
    // cout<<(((ulint)ptr) + align_no - 1)<<" , ";
    // print_2n((((ulint)ptr) + align_no - 1));
    // cout<<~(align_no - 1)<<" , ";
    // print_2n(~align_no - 1);
  return ((void *)((((ulint)ptr) + align_no - 1) & ~(align_no - 1)));
}
int main()
{

    int page_size = 32;

    int * buf2 = (int*)malloc(3*page_size);
    cout<<"buf2 : "<<endl;
    cout<<(ulint)buf2<<" , ";
    print_2n((ulint)buf2);
    cout<<"in ut_align"<<endl;
    ut_align(buf2+1,page_size);
    cout<<"out ut_align"<<endl;
    cout<<buf2<<endl;

    return 0;   
}