#include "../include/system.h"
#include "../include/imfeat.h"
#include "../include/textdetect.h"
#include "../include/util.h"
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
	int ICDAR_2013_end_img_no = 233;//233
	int algo = 1;
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
	G_td.output_mode = DRAW_ER_RECT_IN_IMAGE_AND_SAVE;//SAVE_ER_AS_TEXT_FILE; // or DRAW_ER_RECT_IN_IMAGE_AND_SAVE

	// process each images
	for (int img_id = ICDAR_2013_start_img_no; img_id <= ICDAR_2013_end_img_no; img_id++) {

		float img_resize_ratio = 1.0;
		bool resize = 0;
		CvSize size;

		// load image
		char fn[128];
		sprintf(fn, "%s/img_%d.jpg", G_td.input_path, img_id);
		IplImage *img_rgb = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);
		IplImage *img_yuv = cvCloneImage(img_rgb);
		cvCvtColor(img_rgb, img_yuv, CV_RGB2YUV);
		G_td.img_orig_rgb = img_rgb;
		G_td.img_orig_yuv = img_yuv;

		// resize if needed
		IplImage *img = img_yuv;
		if (img->width > max_width) {
			resize = true;
			img_resize_ratio = max_width*1.0f / img->width;
			size = cvSize(max_width, (int)(img->height*img_resize_ratio));
			IplImage *img_rs = cvCreateImage(size, img->depth, img->nChannels);
			cvResize(img, img_rs);
			img = img_rs;
		} else {
			size = cvGetSize(img);
		}

		// get y,u,v channel images
		IplImage *y = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *u = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1),
				 *v = cvCreateImage(size, IPL_DEPTH_8U, CV_8UC1);
		cvSplit(img, y, u, v, NULL);

		// generate ER candidates
		generate_ER_candidates(y, img_id, 'y', img_resize_ratio, 0, algo);
		generate_ER_candidates(y, img_id, 'y', img_resize_ratio, 1, algo);
		generate_ER_candidates(u, img_id, 'u', img_resize_ratio, 0, algo);
		generate_ER_candidates(u, img_id, 'u', img_resize_ratio, 1, algo);
		generate_ER_candidates(v, img_id, 'v', img_resize_ratio, 0, algo);
		generate_ER_candidates(v, img_id, 'v', img_resize_ratio, 1, algo);

		// free resource
		cvReleaseImage(&y);
		cvReleaseImage(&u);
		cvReleaseImage(&v);
		if (G_td.img_orig_rgb != NULL) 
			cvReleaseImage(&G_td.img_orig_rgb);
		if (G_td.img_orig_yuv != NULL) 
			cvReleaseImage(&G_td.img_orig_yuv);
		if (resize) 
			cvReleaseImage(&img); //img_rs
	}
_done:

	return 0;
}

int ICDAR2013_evaluate_ER_candidates_by_txt_GroundTruth(void)
{
	int ICDAR_2013_start_img_no = 1;
	int ICDAR_2013_end_img_no = 233;//233

	char in_gdtr[MAX_FN_LEN] =  "../../../../../Dataset/ICDAR_2013/SceneTest_GroundTruth_txt";
	char in[MAX_FN_LEN] = "../../../../../TestResult/ICDAR_2013/09.12/Algo3_txt";
	char out[MAX_FN_LEN] = "../../../../../TestResult/ICDAR_2013";

	float recall = 0;

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

		// load ground truth
		char fn_gt[MAX_FN_LEN];
		sprintf(fn_gt, "%s/gt_img_%d.txt", G_td.groundtruth_path, img_id);
		FILE *f_gt = fopen(fn_gt, "r");
		char word[MAX_FN_LEN];
		CvRect r_gt;

		// load input file
		char fn_in[MAX_FN_LEN];
		sprintf(fn_in, "%s/img_%d.txt", G_td.input_path, img_id);
		FILE *f_in = fopen(fn_in, "r");

		// for each rectangle in ground truth
		float recall_sum = 0;
		int recall_no = 0;
		while (!feof(f_gt)) {
			fscanf(f_gt, "%d, %d, %d, %d, %s\n", &r_gt.x, &r_gt.y, &r_gt.width, &r_gt.height, word);

			recall_no++;
			rect_accumulate_start(r_gt);

			// accumulate each rectangle from input data
			fseek(f_in, 0, SEEK_SET);
			while (!feof(f_in)) {
				CvRect r_in;
				float ym, um, vm;
				char chan[5];
				fscanf(f_in, "%d %d %d %d %f %f %f %s\n", &r_in.x, &r_in.y, &r_in.width, &r_in.height, &ym, &um, &vm, &chan);
				
				CvRect inter = rect_intersect(r_gt, r_in);
				if (inter.width + inter.height == 0)
					continue;
				rect_accumulate_rect(inter);
			}

			recall_sum += rect_accumulate_get_percent();
			rect_accumulate_end();
		}

		fclose(f_in);
		fclose(f_gt);

		//printf("img_%d recall:%1.3f\n", img_id, recall_sum/recall_no);
		recall += (recall_sum/recall_no);
	}

	// write output data
	char fn_out[MAX_FN_LEN];
	sprintf(fn_out, "%s/output.txt", G_td.output_path);
	FILE *f_out = fopen(fn_out, "w");
	fprintf(f_out, "Recall Rate of img:%d~%d is %f",
			ICDAR_2013_start_img_no,
			ICDAR_2013_end_img_no,
			recall / (ICDAR_2013_end_img_no-ICDAR_2013_start_img_no+1));
	fclose(f_out);

_done:
	return 0;
}

void main(void) 
{
	//ICDAR2013_generate_ER_candidates();
	ICDAR2013_evaluate_ER_candidates_by_txt_GroundTruth();

}