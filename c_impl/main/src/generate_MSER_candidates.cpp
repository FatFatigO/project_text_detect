
#include "../include/textdetect.h"
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>

using namespace cv;

static const Vec3b bcolors[] =
{
    Vec3b(0,0,255),
    Vec3b(0,128,255),
    Vec3b(0,255,255),
    Vec3b(0,255,0),
    Vec3b(255,128,0),
    Vec3b(255,255,0),
    Vec3b(255,0,0),
    Vec3b(255,0,255),
    Vec3b(255,255,255)
};

/* Draw ER rectangle in original image and save as jpg */
static void draw_ER_rectangle_in_original_image_and_save(vector<vector<Point>> contours)
{
	IplImage *img;
	char fn[128], img_fn[128];

	// check if output image exist
	int file_exist = 0;
	sprintf(img_fn, G_td.output_fn_format, G_td.img_id);
	sprintf(fn, "%s/%s.jpg", G_td.output_path, img_fn);
	if (FILE * file = fopen(fn, "r")) {
        fclose(file);
        file_exist = 1;
    }
	if (file_exist) {
		// load from output path
		img = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);
	} else {
		// save original color image first
		img = cvCloneImage(G_td.img_orig_rgb);
		sprintf(img_fn, G_td.output_fn_format, G_td.img_id);
		sprintf(fn, "%s/%s.jpg", G_td.output_path, img_fn);
		cvSaveImage(fn, img);
	}

	// draw rect
	CvScalar color;
	if (G_td.img_chan == 'y')
		color = CV_RGB(255, 0, 0);
	else if (G_td.img_chan == 'u')
		color = CV_RGB(0, 255, 0);
	else
		color = CV_RGB(0, 0, 255);
	for (int i = (int)contours.size()-1; i >= 0; i--) {
		const vector<Point>& reg = contours[i];
		int r = 0, b = 0, l = G_td.img->width-1, t = G_td.img->height-1;
		for ( int j = 0; j < (int)reg.size(); j++ ) {
			Point pt = reg[j];
			l = min(l, pt.x);
			r = max(r, pt.x);
			t = min(t, pt.y);
			b = max(b, pt.y);
		}
		cvRectangle(img, cvPoint(l*1.0/G_td.img_resize_ratio,t*1.0/G_td.img_resize_ratio), 
						 cvPoint(r*1.0/G_td.img_resize_ratio,t*1.0/G_td.img_resize_ratio), color, 2);
		cvRectangle(img, cvPoint(r*1.0/G_td.img_resize_ratio,t*1.0/G_td.img_resize_ratio), 
						 cvPoint(r*1.0/G_td.img_resize_ratio,b*1.0/G_td.img_resize_ratio), color, 2);
		cvRectangle(img, cvPoint(r*1.0/G_td.img_resize_ratio,b*1.0/G_td.img_resize_ratio), 
						 cvPoint(l*1.0/G_td.img_resize_ratio,b*1.0/G_td.img_resize_ratio), color, 2);
		cvRectangle(img, cvPoint(l*1.0/G_td.img_resize_ratio,b*1.0/G_td.img_resize_ratio), 
						 cvPoint(l*1.0/G_td.img_resize_ratio,t*1.0/G_td.img_resize_ratio), color, 2);
	}
	
	// save image
	sprintf(img_fn, G_td.output_fn_format, G_td.img_id);
	sprintf(fn, "%s/%s.jpg", G_td.output_path, img_fn);
	cvSaveImage(fn, img);
	cvReleaseImage(&img);
}

/* Save ERs as text file */
static void save_ER_as_text_file(vector<vector<Point>> contours)
{
	char fn[64], img_fn[64];
	sprintf(img_fn, G_td.output_fn_format, G_td.img_id);
	sprintf(fn, "%s/%s.txt", G_td.output_path, img_fn);
	FILE *f = fopen(fn, "a");

	Mat yuv = Mat(G_td.img_orig_yuv, 0);

	for (int i = (int)contours.size()-1; i >= 0; i--) {
		const vector<Point>& reg = contours[i];
		int r = 0, b = 0, l = G_td.img->width-1, t = G_td.img->height-1;
		Vec3d sum = 0.0;
		for ( int j = 0; j < (int)reg.size(); j++ ) {
			Point pt = reg[j];
			l = min(l, pt.x);
			r = max(r, pt.x);
			t = min(t, pt.y);
			b = max(b, pt.y);
			// draw mser's with different colors
			//img_yy.at<Vec3b>(pt) = bcolors[i%9];
			sum += yuv.at<Vec3b>(pt);
		}
		CvRect rect = cvRect((int)((l)*1.0/G_td.img_resize_ratio),
						     (int)((t)*1.0/G_td.img_resize_ratio),
						     (int)((r-l+1)*1.0/G_td.img_resize_ratio),
						     (int)((b-t+1)*1.0/G_td.img_resize_ratio));
		//cvSetImageROI(G_td.img_orig_yuv, rect);
		//CvScalar mean = cvAvg(G_td.img_orig_yuv);
		fprintf(f, "%d	%d	%d	%d	%f	%f	%f	%c\n", 
				rect.x, rect.y, rect.width, rect.height,
				sum.val[0]*1.0/reg.size(), sum.val[1]*1.0/reg.size(), sum.val[2]*1.0/reg.size(), G_td.img_chan);
	}

	fclose(f);
}


void generate_MSER_candidates(IplImage *in_img, int img_id, char img_chan, float img_resize_ratio, int text_is_darker)
{
	// save some parameters
	G_td.img = in_img;                             // input image
	G_td.img_chan = img_chan;                      // image channel name
	G_td.img_id = img_id;                          // image id
	G_td.img_resize_ratio = img_resize_ratio;      // resize ratio
	G_td.r.text_is_darker = text_is_darker;        // text is darker than background or not

	// extract MSER into contours
	vector<vector<Point>> contours;
	int delta=5; int min_area=60; int max_area=in_img->width * in_img->height / 5;
	double max_variation=0.25; double min_diversity=0.7;
    int max_evolution=200; double area_threshold=1.01;
    double min_margin=0.003; int edge_blur_size=5;
	MSER mser4 = MSER(delta, min_area, max_area, max_variation, min_diversity,
						max_evolution, area_threshold, min_margin, edge_blur_size);
	mser4(in_img, contours);
	Mat img = Mat(in_img,0);

	/* output results */
	if (G_td.output_mode == DRAW_ER_RECT_IN_IMAGE_AND_SAVE)
		draw_ER_rectangle_in_original_image_and_save(contours);
	else if (G_td.output_mode == SAVE_ER_AS_TEXT_FILE)
		save_ER_as_text_file(contours);
}