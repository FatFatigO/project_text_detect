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
char in_gdtr[MAX_FN_LEN], in[MAX_FN_LEN], out[MAX_FN_LEN];

int ICDAR2013_generate_ER_candidates(void)
{
	int max_width = 1600;

	// process each images
	for (int img_id = G_td.img_start; img_id <= G_td.img_end; img_id++) {

		float img_resize_ratio = 1.0;

		// load image
		char fn[128];
		Mat img_rgb, img_yuv, img_yuv_ok;
		sprintf(fn, "%s/img_%d.jpg", G_td.input_path, img_id);
		img_rgb = imread(fn, CV_LOAD_IMAGE_COLOR);
		if (img_rgb.dims == 0) {
			sprintf(fn, "%s/img_%d.png", G_td.input_path, img_id);
			img_rgb = imread(fn, CV_LOAD_IMAGE_COLOR);
		}
		assert(img_rgb.dims>0);
		cvtColor(img_rgb, img_yuv, CV_RGB2YUV);
		G_td.img_orig_rgb = &img_rgb;
		G_td.img_orig_yuv = &img_yuv;

		// resize if needed
		if (img_yuv.cols > max_width) {
			img_resize_ratio = max_width*1.0f / img_yuv.cols;
			cv::resize(img_yuv, img_yuv_ok, Size(max_width, (int)(img_yuv.rows*img_resize_ratio)));
		} else {
			img_yuv_ok = img_yuv;
		}

		// get y,u,v channel images
		vector<Mat> chans;
		split(img_yuv_ok, chans);
		G_td.img_orig_y = &chans[0];
		G_td.img_orig_u = &chans[1];
		G_td.img_orig_v = &chans[2];

		if (G_td.get_ER_algo == MSER_ORGINAL) {
			// generate MSER candidates
			generate_MSER_candidates(G_td.img_orig_y, img_id, 'y', img_resize_ratio, 0);
			generate_MSER_candidates(G_td.img_orig_y, img_id, 'y', img_resize_ratio, 1);
			generate_MSER_candidates(G_td.img_orig_u, img_id, 'u', img_resize_ratio, 0);
			generate_MSER_candidates(G_td.img_orig_u, img_id, 'u', img_resize_ratio, 1);
			generate_MSER_candidates(G_td.img_orig_v, img_id, 'v', img_resize_ratio, 0);
			generate_MSER_candidates(G_td.img_orig_v, img_id, 'v', img_resize_ratio, 1);
		} else {
			// generate ER candidates
			generate_ER_candidates(G_td.img_orig_y, img_id, 'y', img_resize_ratio, 0);
			generate_ER_candidates(G_td.img_orig_y, img_id, 'y', img_resize_ratio, 1);
			generate_ER_candidates(G_td.img_orig_u, img_id, 'u', img_resize_ratio, 0);
			generate_ER_candidates(G_td.img_orig_u, img_id, 'u', img_resize_ratio, 1);
			generate_ER_candidates(G_td.img_orig_v, img_id, 'v', img_resize_ratio, 0);
			generate_ER_candidates(G_td.img_orig_v, img_id, 'v', img_resize_ratio, 1);
		}
		printf("img %d is done.\n", img_id);
	}

	return 0;
}

int ICDAR2013_evaluate_ER_candidates_by_txt_GroundTruth(void)
{
	double recall = 0;

	// process each images
	for (int img_id = G_td.img_start; img_id <= G_td.img_end; img_id++) {

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
		double recall_sum = 0;
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
	sprintf(fn_out, "%s/Recall_rate_rectangle_level.txt", G_td.output_path);
	FILE *f_out = fopen(fn_out, "w");
	fprintf(f_out, "Recall Rate of img:%d~%d is %f",
			G_td.img_start, G_td.img_end,
			recall / (G_td.img_end-G_td.img_start+1));
	fclose(f_out);

	return 0;
}

int ICDAR2013_evaluate_ER_candidates_by_png_GroundTruth(void)
{
	double recall = 0;

	// process each images
	for (int img_id = G_td.img_start; img_id <= G_td.img_end; img_id++) {

		// load ground truth
		char fn_gt[MAX_FN_LEN];
		sprintf(fn_gt, "%s/gt_img_%d.png", G_td.groundtruth_path, img_id);
		Mat img_bin, img_gray = imread(fn_gt, CV_LOAD_IMAGE_GRAYSCALE);

		// binarize: threshold:254, maxvalue:1, mode:inverted(=>text=1,bg=0)
		threshold(img_gray, img_bin, 254, 1, 1);
		CvRect r_gt = cvRect(0, 0, img_bin.cols, img_bin.rows);

		// load input file
		char fn_in[MAX_FN_LEN];
		sprintf(fn_in, "%s/img_%d.txt", G_td.input_path, img_id);
		FILE *f_in = fopen(fn_in, "r");

		// accumulate each rectangle from input data
		rect_accumulate_start(r_gt, img_bin);
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
		fclose(f_in);

		recall += rect_accumulate_get_percent();
		rect_accumulate_end();
	}

	// write output data
	char fn_out[MAX_FN_LEN];
	sprintf(fn_out, "%s/Recall_rate_pixel_level.txt", G_td.output_path);
	FILE *f_out = fopen(fn_out, "w");
	fprintf(f_out, "Recall Rate of img:%d~%d is %f",
			G_td.img_start, G_td.img_end,
			recall / (G_td.img_end-G_td.img_start+1));
	fclose(f_out);

	return 0;
}

int ICDAR2013_evaluate_ER_candidates_by_gen_stats_from_txt(void)
{
	// write output data
	char fn_out[MAX_FN_LEN];
	sprintf(fn_out, "%s/Statistics.txt", G_td.output_path);
	FILE *f_out = fopen(fn_out, "a");
	fprintf(f_out, "   Y    U    V  Sum\n");

	// process each images
	int total = 0;
	for (int img_id = G_td.img_start; img_id <= G_td.img_end; img_id++) {

		// load input file
		char fn_in[MAX_FN_LEN];
		sprintf(fn_in, "%s/img_%d.txt", G_td.input_path, img_id);
		FILE *f_in = fopen(fn_in, "r");

		// for each rectangle in ground truth
		int line[6] = {0,0,0,0,0,0};
		while (!feof(f_in)) {
			int l,t,w,h, chan = 0;
			double y = 0, u = 0, v = 0;
			fscanf(f_in, "%d %d %d %d %f %f %f %d\n", &l, &t, &w, &h, &y, &u, &v, &chan);
			line[chan]++;
		}
		fclose(f_in);

		// write output data
		fprintf(f_out, "%4d %4d %4d %4d\n", 
				line[0]+line[1],
				line[2]+line[3],
				line[4]+line[5],
				line[0]+line[1]+line[2]+line[3]+line[4]+line[5]);
		total += line[0]+line[1]+line[2]+line[3]+line[4]+line[5];
		
	}
	fprintf(f_out, "Total: %4d regions", total);
	fclose(f_out);

	return 0;
}

int ICDAR2013_random_copy_n_ERs_from_one_to_another_folder()
{
	int n = 2500;

	uchar *used = (uchar *)malloc((G_td.img_end-G_td.img_start+1)*sizeof(uchar));
	memset(used, 0, (G_td.img_end-G_td.img_start+1)*sizeof(uchar));
	int *map = (int *)malloc(n*sizeof(int));
	memset(map, 0, n*sizeof(int));

	RNG rng(0xFFFFFFFF);
	for (int i=0; i<n; i++) {
		int idx = 0;
		do {
			idx = rng.uniform(G_td.img_start, G_td.img_end);
		} while (used[idx] == 1);
		used[idx] = 1;
		map[i] = idx;
	}

	char img_fn[MAX_FN_LEN], fn_src[MAX_FN_LEN], fn_dst[MAX_FN_LEN];
	for (int i=0; i<n; i++) {
		sprintf(img_fn, G_td.output_fn_format, map[i]);
		sprintf(fn_src, "%s/%s.png", G_td.input_path, img_fn);
		sprintf(fn_dst, "%s/%s.png", G_td.output_path, img_fn);
		Mat img = imread(fn_src);
		imwrite(fn_dst, img);
	}

	return 0;
}

int init()
{
	memset(&G_td, 0, sizeof(G_td));

	G_td.global_cnt = 0;//970190;
	G_td.img_start = 1;//301;
	G_td.img_end = 1466742;//233(Test),410(Trian),1466742(TrainERs)

	// Ground truth
	//sprintf(in_gdtr, "../../../../../Dataset/ICDAR_2013/SceneTest_GroundTruth_png");
	sprintf(in_gdtr, "../../../../../Dataset/ICDAR_2013/SceneTest_GroundTruth_txt");

	// Input
	//sprintf(in, "../../../../../Dataset/ICDAR_2013/SceneTest");
	//sprintf(in, "../../../../../Dataset/ICDAR_2013/SceneTrain");
	sprintf(in, "../../../../../TestResult/ICDAR_2013/Train_ER_all");

	// Output
	sprintf(out, "../../../../../TestResult/ICDAR_2013/Train_ER_selected");
	G_td.output_fn_format = "img_%d";

	// Output mode
	G_td.output_mode = SAVE_ER_AS_BIN_PNG;
	//G_td.output_mode = DRAW_ER_RECT_IN_ORIGINAL_IMAGE_AND_SAVE;
	//G_td.output_mode = DRAW_ER_RECT_IN_GNDTRUTH_IMAGE_AND_SAVE;
	//G_td.output_mode = SAVE_ER_AS_TEXT_FILE;

	// Get ER algo
	//G_td.get_ER_algo = MSER_ORGINAL;
	G_td.get_ER_algo = ER_NO_PRUNING;
	//G_td.get_ER_algo = ER_SIZE_VAR_WITH_AR_PENALTY;
	//G_td.get_ER_algo = ER_POSTP_THEN_SIZE_VAR;

	// check if in / out path exists
	struct stat s;
	if ((stat(in_gdtr, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Ground truth input path doesn't exist. Please create it first.");
		return -1;
	}
	if ((stat(in, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Input path doesn't exist. Please create it first.");
		return -1;
	}
	if ((stat(out, &s)==-1) || !S_ISDIR(s.st_mode)) {
		printf("ERR: Output path doesn't exist. Please create it first.");
		return -1;
	}
	G_td.groundtruth_path = in_gdtr;
	G_td.input_path = in;
	G_td.output_path = out;

	return 0;
}

void main(void) 
{
	if (init() == -1) {int c; scanf("%c",&c); return;}

	//ICDAR2013_random_copy_n_ERs_from_one_to_another_folder();
	//ICDAR2013_generate_ER_candidates();
	//ICDAR2013_evaluate_ER_candidates_by_txt_GroundTruth();
	//ICDAR2013_evaluate_ER_candidates_by_png_GroundTruth();
	//ICDAR2013_evaluate_ER_candidates_by_gen_stats_from_txt();
}