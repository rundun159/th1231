#include "LineSegmentation.hpp"
string OUT_PATH2;
vector<Vec3b> LINE_COLORS{Vec3b(100,0,0),Vec3b(200,0,0),Vec3b(0,100,0),Vec3b(0,200,0),Vec3b(0,0,100),Vec3b(0,0,200)};

int line_cnt=0;
float CONTOUR_AVG_HEIGHT_GLO;
float CONTOUR_AVG_WIDTH_GLO;

cv::Mat BINARY_IMG_GLO;

//used
LineSegmentation::LineSegmentation(string path_of_image, string out) 
{
	this->image_path = path_of_image;
	this->color_img = imread(this->image_path, CV_LOAD_IMAGE_COLOR);
	this->grey_img = imread(this->image_path, CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat smoothed_img;
	cv::blur(grey_img, smoothed_img, Size(3, 3), Point(-1, -1));

	cv::threshold(smoothed_img, binary_img, 0.0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	BINARY_IMG_GLO = this->binary_img;

	this->OUT_PATH = out;
	this->contour_avg_height = 0;
	this->contour_avg_width = 0;
	this->word_cnt = 0;
	this->line_avg_height = 0;
	this->not_bracket_line_cnt = 0;
	this->sliced_img_v = vector<cv::Mat>(0);
}

//used
void LineSegmentation::find_contours() {
    cv::Mat img_clone = this->binary_img;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img_clone, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> bound_rect(contours.size() - 1);

    for (size_t i = 0; i < contours.size() - 1; i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
        bound_rect[i] = boundingRect(Mat(contours_poly[i]));
    }
    
	Rect2d rectangle3;
    vector<Rect> merged_rectangles;
    bool is_repeated;
    Mat drawing = this->color_img.clone();

    for (int i = 0; i < bound_rect.size(); i++) 
	{
        is_repeated = false;

        for (int j = i + 1; j < bound_rect.size(); j++) {
            rectangle3 = bound_rect[i] & bound_rect[j];

            if ((rectangle3.area() == bound_rect[i].area()) || (rectangle3.area() == bound_rect[j].area())) {
                is_repeated = true;
                rectangle3 = bound_rect[i] | bound_rect[j];
                Rect2d merged_rectangle(rectangle3.tl().x, rectangle3.tl().y, rectangle3.width, rectangle3.height);

                if (j == bound_rect.size() - 2)
                    merged_rectangles.push_back(merged_rectangle);

                bound_rect[j] = merged_rectangle;
            }
        }
        if (!is_repeated)
            merged_rectangles.push_back(bound_rect[i]);
    }

    for (size_t i = 0; i < merged_rectangles.size(); i++)
    {
        rectangle(drawing, merged_rectangles[i].tl(), merged_rectangles[i].br(), TEST_LINE_COLOR, 2, 8, 0);

		if(!(merged_rectangles[i].width > merged_rectangles[i].height*5.)&&!(merged_rectangles[i].height > merged_rectangles[i].width*5.))
        {
            this->word_cnt++;
            this->contour_avg_height+=merged_rectangles[i].height;
            this->contour_avg_width+=merged_rectangles[i].width;
        }
    }
    this->contour_avg_height/=this->word_cnt;
    this->contour_avg_width/=this->word_cnt;

	CONTOUR_AVG_WIDTH_GLO = this->contour_avg_width;
	CONTOUR_AVG_HEIGHT_GLO = this->contour_avg_height;

    float final_CONTOUR_AVG_HEIGHT=0;
    float final_CONTOUR_AVG_WIDTH=0;
    float final_WORD_CNT=0;
    for (size_t i = 0; i < merged_rectangles.size(); i++)
    {
        all_TH_contours.push_back(new TH_Rect(merged_rectangles[i]));
        if(!all_TH_contours.back()->is_bar&&!all_TH_contours.back()->is_bracket)
        {
            final_CONTOUR_AVG_HEIGHT+=all_TH_contours.back()->rect.height;
            final_CONTOUR_AVG_WIDTH+=all_TH_contours.back()->rect.width;
            final_WORD_CNT++;
        }
        if(this->all_TH_contours.back()->contain_bar_rect)
            this->rect_contain_bar.push_back(this->all_TH_contours.back());
    }

    CONTOUR_AVG_HEIGHT_GLO=final_CONTOUR_AVG_HEIGHT/final_WORD_CNT;
    CONTOUR_AVG_WIDTH_GLO=final_CONTOUR_AVG_WIDTH/final_WORD_CNT;

    this->contours = merged_rectangles;

	return;
}
check_partition * 
LineSegmentation::get_is_fraction(TH_Rect * now_rect)
{
    int start_y = now_rect->rect.y;
    int end_y = start_y + now_rect->rect.height;
    int start_x = now_rect->rect.x;
    int end_x = start_x + now_rect->rect.width;
    float PAR_IS_FRACTION_RATIO_UPPER=2.0;
    float PAR_IS_FRACTION_RATIO_LOWER=0.0;
    float PAR_DIST_UPPER = 1.2;
    float PAR_DIST_LOWER = 0.6;
    int THR_DIST_UPPER = CONTOUR_AVG_HEIGHT_GLO * PAR_DIST_UPPER;
    int THR_DIST_LOWER = CONTOUR_AVG_HEIGHT_GLO * PAR_DIST_LOWER;
    bool found_above= false;
    bool found_below = false;
    int above_bound = max(0,start_y-THR_DIST_UPPER);
    int below_bound = min(this->img_h,end_y+THR_DIST_UPPER);
    int above_y=0 , below_y = 0;
    int above_dist =0, below_dist =0;
    int max_dist=0,min_dist = 0;
    float dist_ratio = 0;
    for(int y=start_y-1;y>=above_bound;y--)
        if(!found_above)
            for(int x=start_x;x<=end_x;x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                {
                    found_above = true;
                    above_y =y;
                    break;
                }

    if(!found_above)
        return new check_partition(false);

    for(int y=end_y+1;y<=below_bound;y++)
        if(!found_below)
            for(int x=start_x;x<=end_x;x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                {
                    found_below = true;
                    below_y =y;
                    break;
                }

    if(!found_below)
        return new check_partition(false);
    
    above_dist = start_y - above_y;
    below_dist = below_y - end_y;

    max_dist = max(above_dist,below_dist);
    min_dist = min(above_dist, below_dist);
    if(min_dist!=0)
        dist_ratio = (float)max_dist/min_dist;
    else
        dist_ratio = 1.0;
    if(max_dist<THR_DIST_LOWER)
        return new check_partition(true, above_y, below_y);
    else if (dist_ratio<PAR_IS_FRACTION_RATIO_UPPER && dist_ratio> PAR_IS_FRACTION_RATIO_LOWER)
    {
        if (max_dist<THR_DIST_UPPER)
            return new check_partition(true, above_y, below_y);
    }
    return new check_partition(false);
}
//used
void LineSegmentation::get_skewed_angle(pair<int,int> mid) //mid.first : H , mid.second : W
{
    int DEGREES=180;
    vector<float> cos_table = vector<float>(181,0);
    vector<float> sin_table = vector<float>(181,0);
    for(int i=0;i<DEGREES;i++)
    {
        cos_table[i]=cos((double)i/DEGREES*CV_PI);
        sin_table[i]=sin((double)i/DEGREES*CV_PI);
    }
    int ADDITIONAL = 100;
    int ADDITIONAL2 = 20;

    Size size = this->binary_img.size();
    int cenx = mid.second;
    int ceny = mid.first;
    int dist_max = -987654321;
    int edge_x[2] = {0,size.width};
    int edge_y[2] = {0,size.height};
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            dist_max = max(dist_max,int(sqrt(pow(cenx-edge_x[i],2)+pow(ceny-edge_y[j],2))));
    int half_dist = dist_max;

    dist_max *=2;
    dist_max += ADDITIONAL;

    vector<vector<int>> votes = vector<vector<int>>(dist_max,vector<int>(DEGREES,0));

    int start_x = this->start_x;
    int end_x = this->end_x;
    int start_y = this->start_y;
    int end_y = this->end_y;

    uint8_t *myData = this->binary_img.data;
    int stride = this->binary_img.step;

	for (int x = start_x; x<end_x - 1; x++)
	{
		for (int y = start_y; y<end_y - 1; y++)
		{
			uint8_t val = myData[y*stride + x];
			if (val == 0)
			{
				for (int t = 0; t<DEGREES; t++)
				{
					double r = (x - cenx)*cos_table[t] + (y - ceny)*sin_table[t];
					double r2;
					r2 = (r + half_dist + ADDITIONAL2);
					votes[int(r2)][t]++;
				}
			}
		}
	}

    int max_votes=-1;
    int max_theta=-1;
    for(int dist=0; dist<dist_max;dist++)
        for(int theta=0;theta<DEGREES;theta++)
            if(votes[dist][theta]>max_votes)
            {
                max_theta=theta;
                max_votes = votes[dist][theta];
            }


	int skewed_angle = max_theta - 90;

	if (skewed_angle < -60 || skewed_angle > 60)
		return;
	else if (skewed_angle >= -2 && skewed_angle <= 2)
		return;

	Mat matRotation = getRotationMatrix2D(Point(mid.second / 2, mid.first / 2), skewed_angle, 1);

	warpAffine(this->color_img, this->color_img, matRotation, this->binary_img.size(), INTER_LINEAR, 1, Scalar());
	warpAffine(this->grey_img, this->grey_img, matRotation, this->binary_img.size(), INTER_LINEAR, 1, Scalar());

	cv::Mat smoothed_img;
	cv::blur(grey_img, smoothed_img, Size(3, 3), Point(-1, -1));

	cv::threshold(smoothed_img, binary_img, 0.0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	BINARY_IMG_GLO = this->binary_img;

	return;
}

//used
void LineSegmentation::rotate_img()
{
	Size size = this->binary_img.size();
	int img_h = size.height, img_w = size.width;
	uint8_t *myData = this->binary_img.data;
	int stride = this->binary_img.step;
	int minX = 987654321, maxX = -987654321, minY = 987654321, maxY = -987654321;
	for (int x = 0; x<img_w; x++)
	{
		for (int y = 0; y<img_h; y++)
		{
			uint8_t val = myData[y*stride + x];
			if (val == 0)
			{
				minX = min(minX, x);
				minY = min(minY, y);
				maxX = max(maxX, x);
				maxY = max(maxY, y);
			}
		}
	}

	this->start_x = minX, this->end_x = maxX, this->start_y = minY, this->end_y = maxY;

	pair<int, int> mid = pair<int, int>(0, 0);
	mid.first = (this->start_x + this->end_x) / 2;
	mid.second = (this->start_y + this->end_y) / 2;

    this->get_skewed_angle(mid);

    return;
}

//used
void LineSegmentation::segment() 
{
    this->find_contours();
    this->divide_contours_into_lines();
	sort(this->all_lines.begin(), this->all_lines.end(), [ ](LineContainer* c1, LineContainer* c2)
	{
		return c1->start_y <= c2->start_y;
	});

    if(this->not_matched.size()!=0)
        this->include_not_matched();
    if(this->all_Bars.size()!=0)
        this->include_dividend();
    this->merge_lines();
    this->update_range();
    this->imwrite_lines();

    return;
}

//used
void LineSegmentation::divide_contours_into_lines()
{
    float PAR_WORD_THR_COMPARE_LINES = 0.5;
    float PAR_WORD_THR_COMPARE_RECTS = 0.85;
    float PAR_INCLUDED_THR = 0.2;

	sort(this->all_TH_contours.begin(), this->all_TH_contours.end(), [ ](TH_Rect* rect1, TH_Rect* rect2)
	{
		if (rect1->rect.height>rect2->rect.height)
			return true;
		else if (rect1->rect.height == rect2->rect.height)
			return false;
		else if (rect1->rect.height<rect2->rect.height)
			return false;
	});

    if(this->all_TH_contours[0]->is_bar)
        this->all_Bars.push_back(this->all_TH_contours[0]);
    if(this->all_TH_contours[0]->is_bracket)
        this->all_Brackets.push_back(this->all_TH_contours[0]);

    this->all_lines.push_back(new LineContainer(this->all_TH_contours[0]));


    this->update_line_avg_height(0,this->all_lines.back(),0);
    for(int idx_rect=1;idx_rect<this->all_TH_contours.size();idx_rect++)
    {
        TH_Rect * now_rect = this->all_TH_contours[idx_rect];
        if(now_rect->is_bar)
            this->all_Bars.push_back(now_rect);
        if(now_rect->is_bracket)
            this->all_Brackets.push_back(now_rect);
        bool included=false;
        bool new_line =false;
        int included_idx = -1;
        float max_joint = -3;
        int max_line_idx =-1;
        for(int idx = 0; idx<this->all_lines.size();idx++)
        {
            float temp_joint=this->all_lines[idx]->get_jointed(now_rect);
            if(max_joint<temp_joint)
            {
                max_line_idx=idx;
                max_joint=temp_joint;
            }
        }
        if(max_joint>PAR_INCLUDED_THR)
            included=true;
        else
        {
            included=false;
			if (now_rect->is_bar)
				new_line = false;
            else if(now_rect->rect.height>=CONTOUR_AVG_HEIGHT_GLO*PAR_WORD_THR_COMPARE_RECTS)
                new_line=true;
            else
				new_line = false;
        }
        if(included)
        {
			int before_height = this->all_lines[max_line_idx]->end_y - this->all_lines[max_line_idx]->start_y;
            this->all_lines[max_line_idx]->push_contours(now_rect);
            this->update_line_avg_height(1,this->all_lines[max_line_idx],before_height);            
        }        
        else if(new_line)
        {
            this->all_lines.push_back(new LineContainer(now_rect));
            this->update_line_avg_height(0,this->all_lines.back(),0);
        }
        else
            this->not_matched.push_back(now_rect);
	}
    return;
}
void  LineSegmentation::merge_two_lines(int idx1, int idx2)
{
    this->all_lines[idx1]->start_y=min(this->all_lines[idx1]->start_y,this->all_lines[idx2]->start_y);
    this->all_lines[idx1]->end_y=max(this->all_lines[idx1]->end_y,this->all_lines[idx2]->end_y);
    this->all_lines[idx1]->word_avg_height=this->all_lines[idx1]->word_avg_height*this->all_lines[idx1]->num_contours+
                        this->all_lines[idx2]->word_avg_height*this->all_lines[idx2]->num_contours;
    this->all_lines[idx1]->word_avg_width=this->all_lines[idx1]->word_avg_width*this->all_lines[idx1]->num_contours+
                        this->all_lines[idx2]->word_avg_width*this->all_lines[idx2]->num_contours;
    this->all_lines[idx1]->num_contours+=this->all_lines[idx2]->num_contours;
    this->all_lines[idx1]->word_avg_height/=this->all_lines[idx1]->num_contours;
    this->all_lines[idx1]->word_avg_width/=this->all_lines[idx1]->num_contours;
    this->all_lines[idx1]->contain_bracket=this->all_lines[idx1]->contain_bracket||this->all_lines[idx2]->contain_bracket;
    this->all_lines[idx1]->contain_bar=this->all_lines[idx1]->contain_bar||this->all_lines[idx2]->contain_bar;
    this->all_lines[idx1]->contours.insert(this->all_lines[idx1]->contours.end(),this->all_lines[idx2]->contours.begin(),this->all_lines[idx2]->contours.end());
    this->all_lines[idx1]->brackets.insert(this->all_lines[idx1]->brackets.end(),this->all_lines[idx2]->brackets.begin(),this->all_lines[idx2]->brackets.end());
    this->all_lines[idx1]->bars.insert(this->all_lines[idx1]->bars.end(),this->all_lines[idx2]->bars.begin(),this->all_lines[idx2]->bars.end()); 

    vector<LineContainer*>::iterator iter = this->all_lines.begin();
    for(int i=0;i<idx2;i++)
        iter++;
    this->all_lines.erase(iter);   
}

bool ret_include_divide_or_not(int mine_dist,int other_dist,int mean_line_h)
{
    float PAR_INCLUDE_DIVIDEND_THR=0.7;
    float PAR_INCLUDE_DIVIDEND_THR2=1.4;
    float PAR_INCLUDE_DIVIDEND_THR3=3;
    float PAR_COMPARED_TO_GLO_H=0.6;
    float PAR_COMPARED_TO_GLO_H2=0.8;
    float PAR_COMPARED_TO_GLO_H3=0.2;
    float ratio=0;
    if(mine_dist!=0&&other_dist!=0)
        ratio = (float)other_dist/mine_dist;

    if(other_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H)
        if(mine_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H)
        {
            if(ratio<PAR_INCLUDE_DIVIDEND_THR3)
                return true;
        }
    if(other_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H2)
        if(mine_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H2)
        {
            if(mine_dist!=0&&other_dist!=0)
            {
                if(ratio>PAR_INCLUDE_DIVIDEND_THR)
                    if(ratio<PAR_INCLUDE_DIVIDEND_THR2)
                        return true;
            }
            else if(mine_dist==0)
            {
                if(other_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H)
                    return true;
            }
            else if(other_dist==0)
            {
                if(mine_dist<CONTOUR_AVG_HEIGHT_GLO*PAR_COMPARED_TO_GLO_H)
                    return true;
            }
        }
    return false;
}

void LineSegmentation::include_dividend()
{
    float PAR_PASS_THR=2.0;  //this bar should not be a fraction bar
    float PAR_IS_MINE_DIST_TO_SHORT=0.3;
    //if the line that has bar has a joint parts,and the ratio is high, then merge it. 
    if(this->all_lines.size()<2)
        return;
    int idx=0;
    check_partition * ret = NULL;
    bool fraction = false;
    bool merge = false;
    bool above = false;
    int cnt = 0;
    for(idx=0;idx<this->all_lines.size();idx++)
    {
        fraction = false;
        if(this->all_lines[idx]->contain_bar)
        {
            for(int i=0;i<this->all_lines[idx]->bars.size();i++)
            {
                //start from here 
                ret = this->get_is_fraction(this->all_lines[idx]->bars[i]);
                fraction = ret->fraction_bar;
                if(fraction)
                    break;
                else
                    delete ret;
            }
        }
        if(fraction)
        {
            bool include_above_line = false;
            bool include_below_line = false;
            if(idx!=0)
            {
                include_above_line = this->all_lines[idx-1]->is_point_included(ret->above_y);
            }
            if (idx != (this->all_lines.size()-1))
            {
                include_below_line = this->all_lines[idx+1]->is_point_included(ret->below_y);                
            }

            if(include_above_line && include_below_line)
            {
                if(!this->all_lines[idx]->merged&&!this->all_lines[idx+1]->merged&&!this->all_lines[idx+2]->merged)
                {
                    idx--;
                    this->merge_two_lines(idx,idx+1);            
                    this->merge_two_lines(idx,idx+1);
                    this->all_lines[idx]->merged=true;    
                }
            }
            else if (include_above_line)
            {
                if(!this->all_lines[idx]->merged&&!this->all_lines[idx-1]->merged)
                {
                    idx--;
                    this->merge_two_lines(idx,idx+1);                                           
                    this->all_lines[idx]->merged=true;    
                }
            }
            else if(include_below_line)
            {
                if(!this->all_lines[idx]->merged&&!this->all_lines[idx+1]->merged)
                {
                    this->merge_two_lines(idx,idx+1);            
                    this->all_lines[idx]->merged=true;    
                }
            }
        }
    }
}

void LineSegmentation::include_not_matched()
{
    float PAR_IS_FRACTION_RATIO_UPPER=1.5;
    float PAR_IS_FRACTION_RATIO_LOWER=0.0;
    float PAR_INCLUDED_THR = 0.4;
    float PAR_DIST_UPPER = 1.5;
    float PAR_DIST_LOWER = 0.6;
    int i;
    int THR_DIST_UPPER = CONTOUR_AVG_HEIGHT_GLO * PAR_DIST_UPPER;
    int THR_DIST_LOWER = CONTOUR_AVG_HEIGHT_GLO * PAR_DIST_LOWER;
    for(i = 0; i<this->not_matched.size();i++)
    {
        //original one
        // for(line_idx=0;line_idx<this->all_lines.size()&&this->not_matched[i]->rect.y>this->all_lines[line_idx]->start_y;line_idx++);
        bool included_to_line_idx=false;    //just put this rect to line_idx
        bool is_fraction_bar = false;       //merge two lines
        bool is_conjugate_bar = false;      //juet put this rect to line_idx
        int line_idx=0;
        bool overlapped=false;
        bool isOverLine = false;

        while(line_idx<this->all_lines.size())
        {
            overlapped=this->all_lines[line_idx]->is_Overlap(this->not_matched[i]);
            if(overlapped)
                break;
            isOverLine = this->not_matched[i]->rect.y<=this->all_lines[line_idx]->start_y;
            if(isOverLine)
                break;
            line_idx++;
        }
        // for(line_idx=0;line_idx<this->all_lines.size()&&this->not_matched[i]->rect.y>this->all_lines[line_idx]->start_y;line_idx++);
        if(overlapped)
        {
            included_to_line_idx=true;
        }
        else if(isOverLine)
        {
            if(line_idx==0)
                is_conjugate_bar=true;
        }
        else
        { //there's nothing below the bar
            included_to_line_idx=true;
            line_idx--;
        }
        if(!overlapped)
        {
            if(this->not_matched[i]->is_bar)
            {
                //if the not matched rect is bar 
                int idx_not_matched=0; //the bar is right above all_lines[line_idx]

                if(!included_to_line_idx&& !is_conjugate_bar)
                {
                    //may be I should get dist not by histogram but by closest pixel
                    int above_closest_y =this->all_lines[line_idx-1]->get_closest_y_from_below_bar(*this->not_matched[i]);
                    int below_closest_y = this->all_lines[line_idx]->get_closest_y_from_above_bar(*this->not_matched[i]);

                    // int above_peak=this->all_lines[line_idx-1]->get_peak_for_bar(*this->not_matched[i]);
                    // int below_peak=this->all_lines[line_idx]->get_peak_for_bar(*this->not_matched[i]);
                    int bar_peak = this->not_matched[i]->rect.y+this->not_matched[i]->rect.height/2;
                    int above_dst = abs(bar_peak-above_closest_y);
                    int below_dst = abs(bar_peak - below_closest_y);
                    int min_dist = min(above_dst,below_dst);
                    int max_dist = max(above_dst,below_dst);                
                    float dist_rate = ((float)max_dist/min_dist); 
                    if(max_dist<THR_DIST_LOWER)
                    {
                        is_fraction_bar=true;
                    }
                    else if(dist_rate<PAR_IS_FRACTION_RATIO_UPPER && dist_rate > PAR_IS_FRACTION_RATIO_LOWER) //this bar is for fraction
                    {
                        if(max_dist<THR_DIST_UPPER)
                            is_fraction_bar=true;
                        else
                            is_conjugate_bar=true;
                    }
                    else                                //this bar is for conjugate bar
                        is_conjugate_bar=true;
                }
            }
            else
            {
                if(line_idx!=0)
                {
                    int above_closest_y =this->all_lines[line_idx-1]->get_closest_y_from_below_bar(*this->not_matched[i]);
                    int below_closest_y = this->all_lines[line_idx]->get_closest_y_from_above_bar(*this->not_matched[i]);
                    int not_matched_peak = this->not_matched[i]->rect.y+this->not_matched[i]->rect.height/2;
                    int above_dst = abs(not_matched_peak-above_closest_y);
                    int below_dst = abs(not_matched_peak - below_closest_y);
                    if(above_dst<below_dst)
                        line_idx--;                
                }
                included_to_line_idx=true;
            }
        }
        if(is_fraction_bar)
        {
            this->merge_two_lines(line_idx-1,line_idx);
            this->all_lines[line_idx-1]->push_contours(this->not_matched[i]);
        }
        else if(included_to_line_idx||is_conjugate_bar || overlapped)
            this->all_lines[line_idx]->push_contours(this->not_matched[i]);
    }
}

int LineContainer::get_closest_y_from_above_bar(const TH_Rect & bar)  //bar is above this line
{
    bool is_overlap=false;
    int rect_start = bar.rect.y;
    int rect_end=bar.rect.y+bar.rect.height;
    if(rect_start>this->start_y&&rect_end<this->end_y)
        is_overlap=true;
    if(rect_end>=this->start_y&&rect_end<=this->end_y)
        is_overlap=true;    
    if(this->start_y<=rect_start&&this->end_y>=rect_start)
        is_overlap=true;
    if(is_overlap)
    {
        for(int y=rect_end+1;y<=this->end_y;y++)
            for(int x=bar.rect.x;x<=(bar.rect.x+bar.rect.width);x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                    return y;
        return this->end_y;
    }
    else
    {
        for(int y=this->start_y;y<=this->end_y;y++)
            for(int x=bar.rect.x;x<=(bar.rect.x+bar.rect.width);x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                    return y;
        return this->end_y;
    }
}

int LineContainer::get_closest_y_from_below_bar(const TH_Rect & bar)
{
    int ret;
    bool is_overlap=false;
    int rect_start = bar.rect.y;
    int rect_end=bar.rect.y+bar.rect.height;
    if(rect_start>this->start_y&&rect_end<this->end_y)
        is_overlap=true;
    if(rect_end>=this->start_y&&rect_end<=this->end_y)
        is_overlap=true;    
    if(this->start_y<=rect_start&&this->end_y>=rect_start)
        is_overlap=true;

    if(is_overlap)
    {
        for(int y=rect_start-1;y>=this->start_y;y--)
            for(int x=bar.rect.x;x<=(bar.rect.x+bar.rect.width);x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                    return y;
        return this->start_y;
    }
    else
    {
        for(int y=this->end_y;y>=this->start_y;y--)
            for(int x=bar.rect.x;x<=(bar.rect.x+bar.rect.width);x++)
                if(BINARY_IMG_GLO.at<uchar>(y,x)==0)
                    return y;
        return this->start_y;
    }
}
bool LineContainer::is_point_included(int y)
{
    if( this->start_y<=y && this->end_y>=y)
        return true;
    else
        return false;
}

void LineContainer::push_contours(TH_Rect * rect)
{
    contours.push_back(rect);
    this->start_y=min(this->start_y,rect->rect.y);
    this->end_y=max(this->end_y,rect->rect.y+rect->rect.height);
    if(!rect->is_bar&&!rect->is_bracket)
    {
        word_avg_height*=num_contours;
        word_avg_width*=num_contours;

        word_avg_height+=rect->rect.height;
        word_avg_width+=rect->rect.width;
        num_contours++;

        word_avg_height/=num_contours;
        word_avg_width/=num_contours;   
    }
    if(rect->is_bar)
    {
        this->bars.push_back(rect);
        this->contain_bar=true;
    }
    if(rect->is_bracket)
    {
        this->brackets.push_back(rect);
        this->contain_bracket=true;
    }
}

void LineSegmentation::update_range()
{
    for(int idx=0;idx<this->all_lines.size();idx++)
    {
        int minY=987654321,maxY=-1,minX=987654321,maxX=-1;
        for(int i=0;i<this->all_lines[idx]->contours.size();i++)
        {
            minY=min(minY,this->all_lines[idx]->contours[i]->rect.y);
            maxY=max(maxY,this->all_lines[idx]->contours[i]->rect.y+this->all_lines[idx]->contours[i]->rect.height);            
            minX=min(minX,this->all_lines[idx]->contours[i]->rect.x);
            maxX=max(maxX,this->all_lines[idx]->contours[i]->rect.x+this->all_lines[idx]->contours[i]->rect.width);            
        }
        this->all_lines[idx]->start_y = minY, this->all_lines[idx]->end_y = maxY, this->all_lines[idx]->start_x = minX, this->all_lines[idx]->end_x = maxX;
    }
}

void LineSegmentation::imwrite_lines()
{
    int cnt=1;
    int PADDING = 60;
    bool TUNIG_TO_COLOR=true;

    for(int idx=0;idx<this->all_lines.size();idx++)
    {
        int minY=this->all_lines[idx]->start_y,maxY=this->all_lines[idx]->end_y,minX=this->all_lines[idx]->start_x,maxX=this->all_lines[idx]->end_x;
        Mat drawing(maxY-minY+PADDING*2,maxX-minX+PADDING*2,CV_8U,255);
        if(TUNIG_TO_COLOR)
        {
            for(int i=0;i<this->all_lines[idx]->contours.size();i++)
            {
                int startY,endY,startX,endX;
                startY = this->all_lines[idx]->contours[i]->rect.y;
                endY = startY+this->all_lines[idx]->contours[i]->rect.height;
                startX = this->all_lines[idx]->contours[i]->rect.x;
                endX = startX+this->all_lines[idx]->contours[i]->rect.width;
                for(int y = startY;y<= min(endY, grey_img.rows-1);y++)
                    for(int x=startX;x<= min(endX, grey_img.cols-1);x++)
                        drawing.at<uchar>(y-minY+PADDING,x-minX+PADDING)=this->grey_img.at<uchar>(y,x);
            }
        }
        else
        {            
            for(int i=0;i<this->all_lines[idx]->contours.size();i++)
            {
                int startY,endY,startX,endX;
                startY = this->all_lines[idx]->contours[i]->rect.y;
                endY = startY+this->all_lines[idx]->contours[i]->rect.height;
                startX = this->all_lines[idx]->contours[i]->rect.x;
                endX = startX+this->all_lines[idx]->contours[i]->rect.width;

                for(int y = startY;y<=endY;y++)
                    for(int x=startX;x<=endX;x++)
                        drawing.at<uchar>(y-minY+PADDING,x-minX+PADDING)=this->binary_img.at<uchar>(y,x);
            }
        }
        cv::imwrite(OUT_PATH+"line"+to_string(idx)+".jpg", drawing);
    }
}

void LineSegmentation::print_lines()
{
    int cnt=1;
    for(int idx=0;idx<this->all_lines.size();idx++)
    {
       
        Mat drawing = this->binary_img.clone();
        for(int i=0;i<this->all_lines[idx]->contours.size();i++)
        {
            rectangle(drawing, this->all_lines[idx]->contours[i]->rect.tl(),this->all_lines[idx]->contours[i]->rect.br(),
             LINE_COLORS[this->all_lines[idx]->colorIdx], 2, 8, 0);
        }
        cv::imwrite(OUT_PATH+"line"+to_string(idx)+".jpg", drawing);
    }
}

void LineSegmentation::print_lines_for_dividened(int call_cnt)
{
    int cnt=1;
    Mat drawing = this->color_img.clone();
    for(int idx=0;idx<this->all_lines.size();idx++)
    {
        for(int i=0;i<this->all_lines[idx]->contours.size();i++)
        {
            rectangle(drawing, this->all_lines[idx]->contours[i]->rect.tl(),this->all_lines[idx]->contours[i]->rect.br(),
             LINE_COLORS[this->all_lines[idx]->colorIdx], 2, 8, 0);
        }
    }
 }

void LineSegmentation::print_lines_for_not_matched(int call_cnt)
{
    int cnt=1;
    Mat drawing = this->color_img.clone();
    for(int idx=0;idx<this->all_lines.size();idx++)
    {
        for(int i=0;i<this->all_lines[idx]->contours.size();i++)
        {
            rectangle(drawing, this->all_lines[idx]->contours[i]->rect.tl(),this->all_lines[idx]->contours[i]->rect.br(),
             LINE_COLORS[this->all_lines[idx]->colorIdx], 2, 8, 0);
        }
    }
}

void LineSegmentation::print_lines(int call_cnt)
{
    int cnt=1;
    Mat drawing = this->color_img.clone();
    for(int idx=0;idx<this->all_lines.size();idx++)
    {
        for(int i=0;i<this->all_lines[idx]->contours.size();i++)
        {
            rectangle(drawing, this->all_lines[idx]->contours[i]->rect.tl(),this->all_lines[idx]->contours[i]->rect.br(),
             LINE_COLORS[this->all_lines[idx]->colorIdx], 2, 8, 0);
        }
    }
}

void LineSegmentation::print_bars()
{
    Mat drawing = this->color_img.clone();
    for(int idx=0;idx<this->all_Bars.size();idx++)
    {
            rectangle(drawing, this->all_Bars[idx]->rect.tl(),this->all_Bars[idx]->rect.br(),
             LINE_COLORS[0], 2, 8, 0);
    }
    cv::imwrite(OUT_PATH+"bars.jpg", drawing);
}

void LineSegmentation::print_brackets()
{
    Mat drawing = this->color_img.clone();
    for(int idx=0;idx<this->all_Brackets.size();idx++)
    {
            rectangle(drawing, this->all_Brackets[idx]->rect.tl(),this->all_Brackets[idx]->rect.br(),
             LINE_COLORS[0], 2, 8, 0);
    }
    cv::imwrite(OUT_PATH+"brakets.jpg", drawing);

}

void LineSegmentation::update_line_avg_height(int mode, LineContainer * newContainer, int before_height)
{
    if(newContainer->contain_bracket)
        return;
    else
    {
        this->line_avg_height*=this->not_bracket_line_cnt;
        this->line_avg_height+=(newContainer->end_y-newContainer->start_y);
        if(mode==0)     //If the line is added, update average value
            this->not_bracket_line_cnt++;
        else if (mode==1)         //if the line is modified
            this->line_avg_height-=before_height;
        this->line_avg_height/=this->not_bracket_line_cnt;
    }
}

void LineContainer::push_contours(TH_Rect * rect, bool is_word)
{
    contours.push_back(rect);
    this->start_y=min(this->start_y,rect->rect.y);
    this->end_y=max(this->end_y,rect->rect.y+rect->rect.height);
    if(!rect->is_bar&&!rect->is_bracket&&is_word)
    {
        word_avg_height*=num_contours;
        word_avg_width*=num_contours;

        word_avg_height+=rect->rect.height;
        word_avg_width+=rect->rect.width;
        num_contours++;

        word_avg_height/=num_contours;
        word_avg_width/=num_contours;       
    }
    if(rect->is_bar)
    {
        this->bars.push_back(rect);
        this->contain_bar=true;
    }
    if(rect->is_bracket)
    {
        this->brackets.push_back(rect);
        this->contain_bracket=true;
    }
}

float LineContainer::get_jointed(TH_Rect* rect)
{
    int sw=0;
	int rect_start = rect->rect.y;
	int rect_end = rect->rect.y + rect->rect.height;
	
	if (rect_start > this->start_y && rect_end < this->end_y)
		sw = 1;
	else if (rect_end >= this->start_y && rect_end <= this->end_y)
		sw = 1;
	else if (this->start_y <= rect_start && this->end_y >= rect_start)
		sw = 1;
	else if (rect_end < this->start_y && this->start_y - rect_end == 0)
		sw = 1;
	else if (rect_start > this->end_y && rect_start - this->end_y == 0)
		sw = 1;

	if (sw == 1)
	{
		if (rect_start >= this->start_y)
		{
			if (rect_end <= this->end_y)
				return 1.;
			else if (this->end_y >= rect_start)
			{
				return (float)(this->end_y - rect_start) / rect->rect.height;
			}
			else
				return 0.;
		}
		else
		{
			if (rect_end >= this->start_y)
			{
				if (rect_end<this->end_y)
					return (float)(rect_end - this->start_y) / rect->rect.height;
				else
					return (float)(this->end_y - this->start_y) / rect->rect.height;
			}
			else
				return 0.;
		}
	}
    else
        return 0.;
}
bool 
LineContainer::is_Overlap(TH_Rect * rect)
{
    int rect_start = rect->rect.y;
    int rect_end=rect->rect.y+rect->rect.height;
    if(rect_start>this->start_y&&rect_end<this->end_y)
        return true;
    if(rect_end>=this->start_y&&rect_end<=this->end_y)
        return true;    
    if(this->start_y<=rect_start&&this->end_y>=rect_start)
        return true;
    return false;
}

LineContainer::LineContainer()
:start_y(987654321),end_y(-1),word_avg_height(0),word_avg_width(0),num_contours(0),last_word_height(0),contain_bar(false),contain_bracket(false),
max_height_contour(NULL),min_height_contour(NULL),max_width_contour(NULL),min_width_contour(NULL), merged(false)
{
    this->colorIdx=(line_cnt++)%LINE_COLORS.size();
}

LineContainer::LineContainer(TH_Rect * rect):
start_y(987654321),end_y(-1),word_avg_height(0),word_avg_width(0),num_contours(0),last_word_height(0),contain_bar(false),contain_bracket(false),
max_height_contour(NULL),min_height_contour(NULL),max_width_contour(NULL),min_width_contour(NULL), merged(false)    
{
    this->push_contours(rect);
    this->colorIdx=(line_cnt++)%LINE_COLORS.size();
}

TH_Rect::TH_Rect():is_bracket(false),is_bar(false),is_small(false), contain_bar_rect(false),bar_position(-1) {}

TH_Rect::TH_Rect(Rect input_rect):rect(input_rect), contain_bar_rect(false),bar_position(-1)
{
    this->is_bracket = this->get_is_bracket();
    this->is_bar=this->get_is_bar();
    this->is_small=false;
    this->get_contain_bar_rect();
}

TH_Rect::TH_Rect(Rect input_rect, bool is_bracket, bool is_bar):rect(input_rect),is_bracket(is_bracket),is_bar(is_bar),is_small(false), contain_bar_rect(false),bar_position(-1) {}

bool TH_Rect::get_is_bar()
{
	float PAR_BAR_THR=5;
	float PAR_BAR_THR2=3;
	float PAR_BAR_AVG_THR=0.8;
	float PAR_BAR_AVG_THR2=0.5;

	if (this->rect.width >= this->rect.height*PAR_BAR_THR && this->rect.width >= CONTOUR_AVG_WIDTH_GLO*PAR_BAR_AVG_THR)
		return true;
	if (this->rect.width >= this->rect.height*PAR_BAR_THR2 && this->rect.height <= CONTOUR_AVG_HEIGHT_GLO*PAR_BAR_AVG_THR2)
		return true;

	return false;
}

bool TH_Rect::get_is_bracket()
{
    float PAR_BRACKET_THR=5;
    float PAR_BRACKET_AVG_THR=1.5;
    float PAR_BRACKET_AVG_THR2=3;
	if (this->rect.height > this->rect.width*PAR_BRACKET_THR && this->rect.height > CONTOUR_AVG_HEIGHT_GLO*PAR_BRACKET_AVG_THR)
		return true;
    else if(this->rect.height > CONTOUR_AVG_HEIGHT_GLO*PAR_BRACKET_AVG_THR2)
		return true;

	return false;
}

void TH_Rect::get_contain_bar_rect()
{
    float CONTAIN_BAR_PAR1=0.08, CONTAIN_BAR_PAR2=0.92;
    this->histogram=vector<int>(this->rect.height,0);
    for(int i=0;i<this->rect.height;i++)
    {
        for(int j=this->rect.x;j<(this->rect.x+this->rect.width);j++)
            if(BINARY_IMG_GLO.at<uchar>(i+this->rect.y,j)==0)
                this->histogram[i]++;
    }
    int max_val=-1,max_pos=-1;
    for(int i=0;i<this->rect.height;i++)
        if(this->histogram[i]>max_val)
        {
            max_val=this->histogram[i];
            max_pos=i;
        }
    if(max_pos<(int)this->rect.height*CONTAIN_BAR_PAR1)
    {
        this->contain_bar_rect=true;
        this->bar_position=1;
    }
    if(max_pos>(int)this->rect.height*CONTAIN_BAR_PAR2)
    {
        this->contain_bar_rect=true;
        this->bar_position=2;
    }
}
void LineSegmentation::merge_lines()
{
    float SPACE_THR_PAR = 7.0/10;
    float SPACE_THR_PAR2 = 5.0/10;
    float H_THR_PAR = 5.0/10;
    float H_THR_PAR2 = 10.0/5;
    float NUM_THR_PAR = 2.5/10;
    float NUM_THR_PAR2 = 10.0/2.5;
    bool do_continue = false;
    int CONTOUR_NUM_THR = 10;
    if(this->all_lines.size()<2)
        return;
    int space_sum =0;
    int line_num = this->all_lines.size();
    int space_num=line_num-1;
    float space_avg =0; 
    vector<int> space_v = vector<int>(line_num,0);
    vector<int> line_group = vector<int>(line_num,-1);
    int last_group_id = 0;
    int space_thr = 0;
    int space_thr2 = 0;
    for(int i=1;i<this->all_lines.size();i++)
    {
        int space = this->all_lines[i]->start_y - this->all_lines[i-1]->end_y;
        space_v[i-1] = space;
        if(space>0)
            space_sum+=space;
        else
            space_num--;
    }
    space_avg = (float)space_sum/space_num;
    space_thr = space_avg * SPACE_THR_PAR;
    space_thr2 = CONTOUR_AVG_HEIGHT_GLO*SPACE_THR_PAR2;
    for(int i=0;i<this->all_lines.size()-1;i++)
    {
        if(space_v[i]<space_thr || space_v[i]<space_thr2)
        {
            int above_line_h = this->all_lines[i]->end_y - this->all_lines[i]->start_y;
            int below_line_h = this->all_lines[i+1]->end_y - this->all_lines[i+1]->start_y;
            int above_line_contour_num  = this->all_lines[i]->contours.size();
            int below_line_contour_num = this->all_lines[i+1]->contours.size();
            bool below_larger = false;
            bool above_larger = false;
            float h_ratio = 0;
            float num_ratio = 0;
            bool merge = false;


            if(below_line_h!=0)
                h_ratio = (float)above_line_h/below_line_h;

 
            if(h_ratio==0)
            {
                last_group_id++;
                continue;
            }
 
            if(h_ratio<=H_THR_PAR)
                below_larger = true;
            else if(h_ratio>=H_THR_PAR2)
                above_larger = true;
            
            if(!below_larger && !above_larger)
            {
                last_group_id++;
                continue;
            }

            num_ratio = (float)above_line_contour_num / below_line_contour_num;

            if(below_larger)
                if(num_ratio<=NUM_THR_PAR)
                {
                    if(above_line_contour_num>CONTOUR_NUM_THR)
                    {
                        last_group_id++;
                        continue;
                    }
                    else
                        merge=true;
                }

            if(above_larger)
                if(num_ratio>=NUM_THR_PAR2)
                {
                    if(below_line_contour_num>CONTOUR_NUM_THR)
                    {
                        last_group_id++;
                        continue;
                    }
                    else
                        merge=true;
                }
            if(!merge)
            {
                last_group_id++;
            }
            else
            {
                if(line_group[i]==-1)
                    line_group[i]=last_group_id;
                line_group[i+1] = last_group_id;                
            }
        }
        else
            last_group_id++;
    }
    vector<int> dir_v(0);
    int last_g = line_group[0];
    int cnt = 1;
    for(int i=1;i<line_num;i++)
    {
        if(line_group[i]==last_g)
            cnt++;
        else
        {
            if(last_g==-1)
                dir_v.push_back(cnt*-1);
            else
                dir_v.push_back(cnt);
            last_g = line_group[i];
            cnt=1;
        }
    }
    if(last_g==-1)
        dir_v.push_back(cnt*-1);
    else
        dir_v.push_back(cnt);
    int line_idx = 0;
    for(int i=0;i<dir_v.size();i++)
    {
        if(dir_v[i]<0)
            line_idx += (dir_v[i]*-1);
        else
        {
            for(int j=0;j<dir_v[i]-1;j++)
                this->merge_two_lines(line_idx,line_idx+1);            
            line_idx++;            
        }
    }
    return;
}
