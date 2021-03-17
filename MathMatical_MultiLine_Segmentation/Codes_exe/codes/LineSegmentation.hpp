#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <map>
#include <cstring>
#include <cmath>
//#include <cv.h>
#include <math.h>
#include <opencv/cv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHUNKS_TO_BE_PROCESSED 5
#define TEST_LINE_COLOR cv::Vec3b(0, 0, 255)
typedef int valley_id;
using namespace cv;
using namespace std;

class LineSegmentation;
class TH_Rect;
class LineContainer;
class check_partition;
extern string OUT_PATH;

class LineSegmentation {
private:
    bool not_primes_arr[100007];
    vector<int> primes;
    int redStart = 20;

public:
    LineSegmentation(string path_of_image, string out);
    void rotate_img();
    void segment();
    int line_h, line_w;
    int start_x, end_x;
    int start_y, end_y;
    int img_w,img_h;
    cv::Mat color_img;
    cv::Mat grey_img;
    cv::Mat grey_img2;
    cv::Mat binary_img;
private:
    string OUT_PATH;
    string image_path;
    vector<Rect> contours;
    int avg_line_height;
    int predicted_line_height;
    int chunk_width;
    vector<LineContainer*> all_lines;
    vector<TH_Rect*> all_TH_contours;
    vector<TH_Rect*> all_Brackets;
    vector<TH_Rect*> all_Bars;
    vector<TH_Rect*> not_matched;
    vector<TH_Rect*> rect_contain_bar;

    void divide_contours_into_lines();
    void include_dividend();
    void include_not_matched();
    void merge_two_lines(int idx1, int idx2);
    float line_avg_height;
    void update_line_avg_height(int mode, LineContainer * newContainer,int before_height);
    float contour_avg_height;
    float contour_avg_width;
    int not_bracket_line_cnt;
    void print_lines();

    void print_lines(int cnt);
    void print_lines_for_dividened(int cnt);
    void print_lines_for_not_matched(int cnt);
    void print_bars();
    void print_brackets();
    int skewed_angle;
    void imwrite_lines();
    void get_skewed_angle(pair<int,int> mid);
    void update_range();
    int word_cnt;
    cv::Mat sliced_img;
    vector<cv::Mat> sliced_img_v;
    void find_contours();
    check_partition * 
    get_is_fraction(TH_Rect * now_rect);
    void merge_lines();
};
class LineContainer{
    public:
        int start_y,end_y, word_avg_height, word_avg_width, num_contours; 
        int start_x, end_x;
        int last_word_height; 
        int colorIdx;
        bool contain_bracket, contain_bar;
        bool merged;
        vector<TH_Rect*> contours;
        vector<TH_Rect*> brackets;
        vector<TH_Rect*> bars;
        Rect * max_height_contour;
        Rect * min_height_contour;
        Rect * max_width_contour;
        Rect * min_width_contour;
        void push_contours(TH_Rect * rect);
        void push_contours(TH_Rect * rect, bool is_word);
        float get_jointed(TH_Rect * rect);
        LineContainer();
        LineContainer(TH_Rect * rect);
        int get_closest_y_from_above_bar(const TH_Rect & bar);  //bar is above this bar
        int get_closest_y_from_below_bar(const TH_Rect & bar); //bar is below this bar
        bool is_point_included(int y);
        bool is_Overlap(TH_Rect *);
};
class TH_Rect{
   public:
        Rect rect;
        bool is_bracket, is_bar, is_small, contain_bar_rect;
        int bar_position;
        vector<int> histogram;
        TH_Rect();
        TH_Rect(Rect input_rect);
        TH_Rect(Rect input_rect, bool is_bracket, bool is_bar);
        bool get_is_bar();
        bool get_is_bracket();
        void get_contain_bar_rect();
};
class check_partition
{
    public:
    bool fraction_bar;
    int above_y;
    int below_y;
    check_partition(bool fraction_bar_):fraction_bar(fraction_bar_) {}
    check_partition(bool fb_, int ay, int by):fraction_bar(fb_),above_y(ay),below_y(by) {}
};
