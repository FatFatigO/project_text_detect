
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
using namespace cv;

CvRect rect_intersect(CvRect r1, CvRect r2) 
{ 
    CvRect intersection; 

    // find overlapping region 
    intersection.x = (r1.x < r2.x) ? r2.x : r1.x; 
    intersection.y = (r1.y < r2.y) ? r2.y : r1.y; 
    intersection.width = (r1.x + r1.width < r2.x + r2.width) ? 
        r1.x + r1.width : r2.x + r2.width; 
    intersection.width -= intersection.x; 
    intersection.height = (r1.y + r1.height < r2.y + r2.height) ? 
        r1.y + r1.height : r2.y + r2.height; 
    intersection.height -= intersection.y; 

    // check for non-overlapping regions 
    if ((intersection.width <= 0) || (intersection.height <= 0)) { 
        intersection = cvRect(0, 0, 0, 0); 
    } 

    return intersection; 
}

Mat area, area_roi;
int area_x, area_y;
void rect_accumulate_start(CvRect r)
{
	area = Mat::zeros(r.height, r.width, CV_8UC1);
	area_x = r.x;
	area_y = r.y;
}

void rect_accumulate_start(CvRect r, Mat roi)
{
	area_roi = roi;
	area = Mat::zeros(r.height, r.width, CV_8UC1);
	area_x = r.x;
	area_y = r.y;
}

void rect_accumulate_end(void)
{
#if 0
	// show accumulated pixels
	Mat out;
	resize(area*255,out,Size(300,450));
	imshow("area*255",out);
	waitKey(0);
	// show ground truth pixels
	resize(area_roi*255,out,Size(300,450));
	imshow("area_roi*255",out);
	waitKey(0);
#endif
	area.release();
}

void rect_accumulate_rect(CvRect r)
{
	int x = r.x - area_x;
	int y = r.y - area_y;
	if (area_roi.dims > 0)
		area.colRange(x,x+r.width).rowRange(y,y+r.height) |= area_roi.colRange(x,x+r.width).rowRange(y,y+r.height);
	else
		area.colRange(x,x+r.width).rowRange(y,y+r.height) = 1;
}

double rect_accumulate_get_percent(void)
{
	float b, a = (sum(area).val[0]);
	
	if (area_roi.dims > 0)
		b = sum(area_roi).val[0];
	else
		b = area.size().area();
	return (a/b);
}

void test_rect_accumulate(void) 
{
	Mat roi =  Mat::zeros(10, 10, CV_8U);
	roi.at<uchar>(Point(4,4)) = 1;
	roi.at<uchar>(Point(4,5)) = 1;
	roi.at<uchar>(Point(5,4)) = 1;
	roi.at<uchar>(Point(5,5)) = 1;
	rect_accumulate_start(cvRect(5,5,10,10), roi);
	rect_accumulate_rect(cvRect(10,5,5,5));
	rect_accumulate_rect(cvRect(5,10,5,5));
	rect_accumulate_rect(cvRect(7,7,5,5));
	float a = rect_accumulate_get_percent();
	rect_accumulate_end();
}