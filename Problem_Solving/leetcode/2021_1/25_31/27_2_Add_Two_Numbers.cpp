#include <iostream>
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    static ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int sum = l1->val+l2->val;
        int carry = sum/10;
        bool finished = false;
        sum %= 10;
        ListNode * ret = new ListNode(sum);
        ListNode * back = ret;
        ListNode * longer;
        l1 = l1->next;
        l2 = l2->next;
        while(l1 != nullptr && l2!=nullptr)
        {
            sum = l1->val + l2->val + carry;
            carry = sum/10;
            sum %= 10;
            back->next = new ListNode(sum);
            back = back->next;
            l1 = l1->next, l2=l2->next;
        }
        if(l1 == nullptr && l2 == nullptr)
        {
            if(carry!=0)
            {
                back->next = new ListNode(carry);              
                back = back->next;  
            }
            finished=true;
        }
        else if(l1!=nullptr)
            longer = l1;
        else if(l2!=nullptr)
            longer = l2;
        if(!finished)
        {
            if (carry!=0)
            {
                while(longer!=nullptr)
                {
                    sum = longer->val + carry;
                    carry = sum/10;
                    sum %= 10;
                    back->next = new ListNode(sum);
                    back = back->next;
                    longer = longer->next;
                }           
                if(carry!=0)
                {
                    back->next = new ListNode(carry);              
                    back = back->next;  
                }            
            }
            else
            {
                while(longer!=nullptr)
                {
                    back->next = longer;
                    back = back->next;
                    longer = longer->next;
                }
            }
        }
        return ret;
    }
};

int main()
{
    ListNode * l1, * l2;
    l1 = new ListNode(2);
    l1->next = new ListNode(4);
    l1->next->next = new ListNode(3);

    l2 = new ListNode(5);
    l2->next = new ListNode(6);
    l2->next->next = new ListNode(4);

    ListNode * ret = Solution::addTwoNumbers(l1,l2);

    return 0;
}