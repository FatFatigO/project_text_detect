#include "../include/system.h"
#include "../include/imfeat.h"
#include "../include/textdetect.h"
#include <iostream>
#include <sys/stat.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv/ml.h"

using namespace std;
using namespace cv;

/* global variable */
G_textdetect_t G_td;

int ICDAR2013_generate_ER_candidates(void)
{
	int ICDAR_2013_start_img_no = 1;
	int ICDAR_2013_end_img_no =1;//233
	int algo = 3;
	int max_width = 1600;
	char in[100] =  "../../../../../Dataset/ICDAR_2013/SceneTest";
	//char out[100] = "../../../../../../../LargeFiles/ICDAR_2013"
	char out[100] = "../../../../../TestResult/ICDAR_2013";
	char out_fn_format[100] = "img_%d";

	// check if in / out path exists
	struct stat s;
	if ((stat(in, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Input path doesn't exist. Please create it first.");
		goto _done;
	}
	if ((stat(out, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Output path doesn't exist. Please create it first.");
		goto _done;
	}
	G_td.input_path = in;
	G_td.output_path = out;
	G_td.output_fn_format = out_fn_format;
	G_td.output_mode = 0;

	// process each images
	for (int img_id = ICDAR_2013_start_img_no; img_id <= ICDAR_2013_end_img_no; img_id++) {

		float img_resize_ratio = 1.0;
		CvSize size;

		// load image
		char fn[128];
		sprintf(fn, "%s/img_%d.jpg", G_td.input_path, img_id);
		IplImage *img = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);

		// resize if needed
		if (img->width > max_width) {
			img_resize_ratio = max_width*1.0 / img->width;
			size = cvSize(max_width, (int)img->height*img_resize_ratio);
			IplImage *img_rs = cvCreateImage(size, img->depth, img->nChannels);
			cvResize(img, img_rs);
			cvReleaseImage(&img);
			img = img_rs;
		} else {
			size = cvGetSize(img);
		}

		// get y,u,v channel images
		IplImage *y = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *u = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *v = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1);
		cvSplit(img, y, u, v, NULL);
		cvReleaseImage(&img);
		Mat yy = Mat(y,0);
		Mat uu = Mat(u,0);
		Mat vv = Mat(v,0);

		// generate ER candidates
		generate_ER_candidates(&yy, img_id, 'y', img_resize_ratio, 0, algo);
		generate_ER_candidates(&yy, img_id, 'y', img_resize_ratio, 1, algo);
		generate_ER_candidates(&uu, img_id, 'u', img_resize_ratio, 0, algo);
		generate_ER_candidates(&uu, img_id, 'u', img_resize_ratio, 1, algo);
		generate_ER_candidates(&vv, img_id, 'v', img_resize_ratio, 0, algo);
		generate_ER_candidates(&vv, img_id, 'v', img_resize_ratio, 1, algo);

		// free resource
		yy.release();
		uu.release();
		vv.release();
		cvReleaseImage(&y);
		cvReleaseImage(&u);
		cvReleaseImage(&v);
	}
_done:

	return 0;
}

#if 0
int ICDAR2013_evaluate_ER_candidates(void)
{
	int ICDAR_2013_start_img_no = 1;
	int ICDAR_2013_end_img_no = 233;//233

	char in_gdtr[100] =  "../../../../../Dataset/ICDAR_2013/SceneTest_GroundTruth";
	char in[100] = "../../../../../TestResult/ICDAR_2013";
	char out[100] = "../../../../../TestResult/ICDAR_2013";

	// check if in / out path exists
	struct stat s;
	if ((stat(in_gdtr, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Ground truth input path doesn't exist. Please create it first.");
		goto _done;
	}
	if ((stat(in, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Input path doesn't exist. Please create it first.");
		goto _done;
	}
	if ((stat(out, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Output path doesn't exist. Please create it first.");
		goto _done;
	}
	G_td.groundtruth_path = in_gdtr;
	G_td.input_path = in;
	G_td.output_path = out;
	G_td.output_fn_format = "img_%d";

	// process each images
	for (int img_id = ICDAR_2013_start_img_no; img_id <= ICDAR_2013_end_img_no; img_id++) {

		float img_resize_ratio = 1.0;
		CvSize size;

		// load image
		char fn[128];
		sprintf(fn, "%s/img_%d.jpg", G_td.input_path, img_id);
		IplImage *img = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);

		// resize if needed
		if (img->width > max_width) {
			img_resize_ratio = max_width*1.0 / img->width;
			size = cvSize(max_width, (int)img->height*img_resize_ratio);
			IplImage *img_rs = cvCreateImage(size, img->depth, img->nChannels);
			cvResize(img, img_rs);
			cvReleaseImage(&img);
			img = img_rs;
		} else {
			size = cvGetSize(img);
		}

		// get y,u,v channel images
		IplImage *y = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *u = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *v = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1);
		cvSplit(img, y, u, v, NULL);
		cvReleaseImage(&img);
		Mat yy = Mat(y,0);
		Mat uu = Mat(u,0);
		Mat vv = Mat(v,0);

		// generate ER candidates
		generate_ER_candidates(&yy, img_id, 'y', img_resize_ratio, 0, algo);
		generate_ER_candidates(&yy, img_id, 'y', img_resize_ratio, 1, algo);
		generate_ER_candidates(&uu, img_id, 'u', img_resize_ratio, 0, algo);
		generate_ER_candidates(&uu, img_id, 'u', img_resize_ratio, 1, algo);
		generate_ER_candidates(&vv, img_id, 'v', img_resize_ratio, 0, algo);
		generate_ER_candidates(&vv, img_id, 'v', img_resize_ratio, 1, algo);

		// free resource
		yy.release();
		uu.release();
		vv.release();
		cvReleaseImage(&y);
		cvReleaseImage(&u);
		cvReleaseImage(&v);
	}
_done:

	return 0;
}
#endif

void main(void) {

	ICDAR2013_generate_ER_candidates();

	//ICDAR2013_evaulate_ER_candidates();

}