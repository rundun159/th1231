#include <iostream>
#include <vector>
#define MAX_RANGE 1000
using namespace std;

 // This is the custom function interface.
 // You should not implement it, or speculate about its implementation
class CustomFunction {
public:
    // Returns f(x, y) for any given positive integers x and y.
    // Note that f(x, y) is increasing with respect to both x and y.
    // i.e. f(x, y) < f(x + 1, y), f(x, y) < f(x, y + 1)
    int f(int x, int y);
};
class Point
{
    public:
    int x,y;
    Point(int _x, int _y):x(_x),y(_y){}
};
class Range
{
    public:
    int x1,x2,y1,y2;
    Point mid;
    Range(int _x1,int _x2, int _y1, int _y2):x1(_x1),x2(_x2),y1(_y1),y2(_y2){}
    void get_mid()
    {
        mid = Point((x1+x2)/2,(y1+y2)/2);
    }
    Point ret_mid()
    {
        return mid;
    }
    Range * ret_range1(){
        return new Range(x1,mid.x-1,y1,mid.y-1) ;
    }
    Range * ret_range2(){
        return new Range(x1,mid.x-1,mid.y+1,y2) ;
    }
    Range * ret_range3(){
        return new Range(mid.x+1,x2,y1,mid.y-1) ;
    }
    Range * ret_range4(){
        return new Range(mid.x+1,x2,mid.y+1,y2) ;
    }
    Range * ret_range5(){
        return new Range(x1,mid.x-1,mid.y,mid.y) ;
    }
    Range * ret_range6(){
        return new Range(mid.x,mid.x,mid.y+1,y2) ;
    }
    Range * ret_range7(){
        return new Range(mid.x+1,x2,mid.y,mid.y) ;
    }
    Range * ret_range8(){
        return new Range(mid.x,mid.x,y1,mid.y-1) ;
    }
    bool is_range()
    {
        if(x1>=1&&x1<=MAX_RANGE)
            if(x2>=1&&x2<=MAX_RANGE)
                if(x1<=x2)
                    if(y1>=1&&y1<=MAX_RANGE)
                        if(y2>=1&&y2<=MAX_RANGE)
                            if(y1<=y2)
                                return true;
        return false;
    }    
};
void get_ranges(CustomFunction & CustomFunction, int z, vector<Point> & point_list, Range * range)
{
    if(!range->is_range())
        return;
    else
    {
        range->get_mid();
        int mid_f = CustomFunction.f(range->mid.x,range->mid.y);
        if(mid_f<z)
        {
            vector<Range*> next_ranges(5,nullptr);
            next_ranges[0] = range->ret_range2();
            next_ranges[1] = range->ret_range3();
            next_ranges[2] = range->ret_range4();
            next_ranges[3] = range->ret_range6();
            next_ranges[4] = range->ret_range7();
            delete(range);
            for(int i=0;i<5;i++)
            {
                get_ranges(CustomFunction,z,point_list,next_ranges[i]);
            }
        }
        else if(mid_f==z)
        {
            point_list.push_back(range->mid);
            delete(range);
        }
        else
        {
            vector<Range*> next_ranges(5,nullptr);
            next_ranges[0] = range->ret_range1();
            next_ranges[1] = range->ret_range2();
            next_ranges[2] = range->ret_range3();
            next_ranges[3] = range->ret_range5();
            next_ranges[4] = range->ret_range8();
            delete(range);
            for(int i=0;i<5;i++)
            {
                get_ranges(CustomFunction,z,point_list,next_ranges[i]);
            }
        }        
    }    
}
class Solution {
public:
    vector<vector<int>> findSolution(CustomFunction& customfunction, int z) {
        vector<Point> ret;
        vector<vector<int>> ret_int;
        Range*start = new Range(1,1,MAX_RANGE,MAX_RANGE);
        get_ranges(customfunction, z, ret, start);
        ret_int = vector<vector<int>>(ret.size(),vector<int>(2,0));
        for(int i=0;i<ret.size();i++)
        {
            ret_int[i][0]=ret[i].x;
            ret_int[i][1]=ret[i].y;
        }
        return ret_int;
    }
};