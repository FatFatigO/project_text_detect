
#include "../include/system.h"
#include "../include/imfeat.h"
#include "../include/textdetect.h"
//#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/*
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/imgproc/imgproc.hpp>
*/
using namespace std;
using namespace cv;

#define printff if(0)printf //printf

/* global variable */
G_textdetect_t G_td;

void calc_incremental_feature_multi(int no_fr, int *feat_id_fr, int feat_id_to)
{
	// init "from" param
	int *pt_fr_start_idx = NULL, *pt_fr_size = NULL;
	p4_t *feat_fr_BB = NULL;
	p1_t *feat_fr_PR = NULL, *feat_fr_EN = NULL, *feat_fr_HC = NULL;
	assert(no_fr>=0);
	if (no_fr > 0) {
		pt_fr_start_idx = (int *)malloc(no_fr*sizeof(int));
		pt_fr_size = (int *)malloc(no_fr*sizeof(int));
		feat_fr_BB = (p4_t *)malloc(no_fr*sizeof(p4_t));
		feat_fr_PR = (p1_t *)malloc(no_fr*sizeof(p1_t));
		feat_fr_EN = (p1_t *)malloc(no_fr*sizeof(p1_t));
		feat_fr_HC = (p1_t *)malloc(no_fr*sizeof(p1_t));
		for (int i=0; i<no_fr; i++) {
			memcpy(&feat_fr_BB[i], &G_td.featraw[feat_id_fr[i]].BB, sizeof(p4_t));
			memcpy(&feat_fr_PR[i], &G_td.featraw[feat_id_fr[i]].PR, sizeof(p1_t));
			memcpy(&feat_fr_EN[i], &G_td.featraw[feat_id_fr[i]].EN, sizeof(p1_t));
			memcpy(&feat_fr_HC[i], &G_td.featraw[feat_id_fr[i]].HC, sizeof(p1_t));
			feat_fr_HC[i].val[0] = (u32)G_td.featraw[feat_id_fr[i]].HC_buf;
			pt_fr_start_idx[i] = G_td.ERs[feat_id_fr[i]].ER_head->pt_order;
			pt_fr_size[i] = G_td.ERs[feat_id_fr[i]].ER_size;
		}
	}

	// init "to" param
	G_td.featraw[feat_id_to].HC_buf = (int *)malloc(G_td.img->rows*sizeof(int));
	memset((void *)G_td.featraw[feat_id_to].HC_buf, 0, G_td.img->rows*sizeof(int));
	featraw_t *feat_to = &G_td.featraw[feat_id_to];
	G_td.featraw[feat_id_to].HC.val[0] = (u32)G_td.featraw[feat_id_to].HC_buf;
	int pt_to_start_idx = G_td.ERs[feat_id_to].ER_head->pt_order;
	int pt_to_size = G_td.ERs[feat_id_to].ER_size;

	// init "pt" param
	p8_t pt;
	pt.val[0] = (u32)G_td.pts;
	pt.val[1] = G_td.img->rows;
	pt.val[2] = G_td.img->cols;
	pt.val[3] = no_fr;
	pt.val[4] = (u32)pt_fr_start_idx;
	pt.val[5] = (u32)pt_fr_size;
	pt.val[6] = pt_to_start_idx;
	pt.val[7] = pt_to_size;

	// bounding box
	get_BoundingBox(IN &pt, IN feat_fr_BB, OUT &feat_to->BB);

	// perimeter
	get_Perimeter(IN &pt, IN feat_fr_PR, OUT &feat_to->PR);

	// euler no
	get_EulerNo(IN &pt, IN feat_fr_EN, OUT &feat_to->EN);

	// horizontal crossig
	get_HzCrossing(IN &pt, IN feat_fr_HC, OUT &feat_to->HC);

	printff("[%d] val %d size %d\n out feature : \n BB:%d,%d,%d,%d\n PR:%d\n EN:%d\n HC:%d,%d,%d,%d\n", 
		feat_id_to, G_td.ERs[feat_id_to].ER_val, G_td.ERs[feat_id_to].ER_size,
		feat_to->BB.val[0], feat_to->BB.val[1], feat_to->BB.val[2], feat_to->BB.val[3],
		feat_to->PR.val[0], feat_to->EN.val[0], *(((int*)feat_to->HC.val[0])+0), *(((int*)feat_to->HC.val[0])+1), *(((int*)feat_to->HC.val[0])+2), *(((int*)feat_to->HC.val[0])+3));

	// relase memory
	if (no_fr > 0) {
		free(pt_fr_start_idx);
		free(pt_fr_size);
		free(feat_fr_BB);
		free(feat_fr_PR);
		free(feat_fr_EN);
		free(feat_fr_HC);
	}
}

/*
procedure LINEAR-REDUCTION(T)
2: if nchildren[T] = 0 then
3:		return T
4: else if nchildren[T] = 1 then
5:		c = LINEAR-REDUCTION(child[T])
6:		if var[T]  var[c] then
7:			link-children(T, children[c])
8:			return T
9:		else
10:			return c
11:		end if
12:else % nchildren[T]  2
13:		for each c 2 children[T] do
14:			link-children(T, LINEAR-REDUCTION(c))
15:		end for
16:		return T
17:end if
end procedure
*/

ER_t *linear_reduction(ER_t *T)
{
	int a = T->ER_id;
	if (T->ER_firstChild ==-1) {
		// has no child
		printff("[%d] has no child, return %d\n",a,T->ER_id);

		/* ---------------------------- START ---------------------------------*/
		// <====== Feature extraction (in/crementally computed from Null => T)
		calc_incremental_feature_multi(0, NULL, T->ER_id);
		/* ----------------------------- END ----------------------------------*/

		return T;
	} else if (T->to_firstChild->ER_nextSibling==-1) {
		// has only one child
		printff("[%d] %d has one child\n",a,T->ER_id);
		printff("[%d] %d go linear reduction \n",a,T->to_firstChild->ER_id);
		ER_t *c = linear_reduction(T->to_firstChild);
		printff("[%d] %d returned from linear reduction \n",a,c->ER_id);
		if (1){//(T->ER_val<c->ER_val) {       // put selection algo here
			// link T to its new child "c"
			c->ER_nextSibling = c->to_parent->ER_nextSibling;
			c->to_nextSibling = c->to_parent->to_nextSibling;
			if (c->to_nextSibling) {
				c->to_nextSibling->to_prevSibling = c;
				c->to_nextSibling->ER_prevSibling = c->ER_id;
			}
			c->ER_prevSibling = c->to_parent->ER_prevSibling;
			c->to_prevSibling = c->to_parent->to_prevSibling;
			if (c->to_prevSibling) {
				c->to_prevSibling->to_nextSibling = c;
				c->to_prevSibling->ER_nextSibling = c->ER_id;
			}
			c->ER_parent = T->ER_id;
			c->to_parent = T;
			printff("[%d] link %d to its new child %d\n",a,T->ER_id,c->ER_id);
			printff("[%d] return %d \n",a,T->ER_id);

			/* ---------------------------- START ---------------------------------*/
			// Feature extraction (incrementally computed from T->firstChild => T)
			int *fr_id = &T->to_firstChild->ER_id;
			calc_incremental_feature_multi(1, fr_id, T->ER_id);
			/* ----------------------------- END ----------------------------------*/

			return T; // T is better
		} else {
			printff("[%d] return %d \n",a,c->ER_id);
			return c; // c is better
		}
	} else {
		// has more than two children
		printff("[%d] %d has more than two child\n",a,T->ER_id);
		ER_t *c = T->to_firstChild;
		int childNo = 0;
		while (c) {
			printff("[%d] %d go linear reduction \n",a,c->ER_id);

			//G_td.featraw[c->ER_id].HC_vec = (int *)malloc(G_td.img->rows*sizeof(int));
			ER_t *d = linear_reduction(c);

			printff("[%d] %d returned from linear reduction \n",a,d->ER_id);
			// link T to its new child "d"
			childNo++;
			if ((T->ER_firstChild!=d->ER_id) && (c->ER_id!=d->ER_id)) {
				if (childNo==1) {
					T->ER_firstChild = d->ER_id;
					T->to_firstChild = d;
				}
				d->ER_nextSibling = d->to_parent->ER_nextSibling;
				d->to_nextSibling = d->to_parent->to_nextSibling;
				if (d->to_nextSibling) {
					d->to_nextSibling->to_prevSibling = d;
					d->to_nextSibling->ER_prevSibling = d->ER_id;
				}
				d->ER_prevSibling = d->to_parent->ER_prevSibling;
				d->to_prevSibling = d->to_parent->to_prevSibling;
				if (d->to_prevSibling) {
					d->to_prevSibling->to_nextSibling = d;
					d->to_prevSibling->ER_nextSibling = d->ER_id;
				}
				d->ER_parent = T->ER_id;
				d->to_parent = T;
			}
			printff("[%d] link %d to its new child %d\n",a,T->ER_id,d->ER_id);
			// next T's child
			c = c->to_nextSibling;
		}
		printff("[%d] return %d \n",a,T->ER_id);

		/* ---------------------------- START ---------------------------------*/
		// Feature extraction (incrementally computed from T->firstChild => T)
		int *fr_id = (int *)malloc(childNo*sizeof(int));
		c = T->to_firstChild;
		for (int i=0; i<childNo; i++) {
			fr_id[i] = c->ER_id;
			c = c->to_nextSibling;
		}
		calc_incremental_feature_multi(childNo, fr_id, T->ER_id);
		free(fr_id);
		/* ----------------------------- END ----------------------------------*/

		return T;
	}
}

void get_ER_candidates(void)
{
	linear_reduction(&G_td.ERs[G_td.ER_no-1]);



}


void main_sample_2(void)
{
	string path_prefix = "../../";
	string path_filelist = path_prefix + "../../../Dataset/MSRA-TD500/test/parsed_filenames.txt";
	ifstream fin(path_filelist);
	int img_no;

	fin >> img_no;
	for (int ii=1; ii<=img_no; ii++)
	{
		clock_t tStart = clock();
		
		// get image
		char path_img[128];
		fin >> path_img;
		Mat img;
		if (0) {
			img = imread(path_prefix + path_img, CV_LOAD_IMAGE_GRAYSCALE);
		} else {
			double percent = 25;
			IplImage *src = cvLoadImage((path_prefix + path_img).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			IplImage *dst = cvCreateImage(cvSize((int)((src->width*percent)/100) , (int)((src->height*percent)/100) ), src->depth, src->nChannels);
			cvResize(src, dst, CV_INTER_LINEAR);
			img = dst;
			//cvNamedWindow("a");
			//cvShowImage("a", dst);
			//cvWaitKey(0);
		}
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock();

		// get ERs
		ER_t *ERs = (ER_t *)malloc(img.rows*img.cols*sizeof(ERs[0]));
		LinkedPoint *pts = (LinkedPoint*)malloc((img.rows*img.cols+1)*sizeof(pts[0]));
		int ER_no = get_ERs(img.data, img.rows, img.cols, 0/*2:see debug msg*/, ERs, pts);
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock(); 

		// assign some global variables
		G_td.img = &img;
		G_td.ERs = ERs;
		G_td.ER_no = ER_no;
		G_td.pts = pts;

		// prepare for getting ER candidates
		G_td.featraw = (featraw_t *)malloc(ER_no*sizeof(featraw_t));
		memset(G_td.featraw, 0, ER_no*sizeof(featraw_t));

		// get ER candidates
		get_ER_candidates();
		free(ERs);
		free(pts);
		for (int i=0; i<ER_no; i++) {
			if (G_td.featraw[i].HC_buf)
				free(G_td.featraw[i].HC_buf);
		}
		free(G_td.featraw);
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock();

		printf("[%d] main_sample_2 test is good\n", ii);
		char ch;
		scanf("%c", &ch);
	}
}


int main_sample_1(void) 
{
	// get image
	int img_cols = 4;
	int img_rows = 4;
	u8* img_data = (u8*)malloc(img_cols*img_rows*sizeof(u8));
	u8* img_ptr = img_data;
	img_ptr[0] =  1; img_ptr[1] =  2; img_ptr[2] =  3; img_ptr[3] =  1;
	img_ptr[4] =  2; img_ptr[5] =  2; img_ptr[6] =  3; img_ptr[7] =  2;
	img_ptr[8] =  3; img_ptr[9] =  3; img_ptr[10] = 3; img_ptr[11] = 2;
	img_ptr[12] = 1; img_ptr[13] = 2; img_ptr[14] = 3; img_ptr[15] = 4;
	Mat img;
	img.rows = img_rows;
	img.cols = img_cols;

	// get ERs
	ER_t* ERs = (ER_t *)malloc(img_rows*img_cols*sizeof(ERs[0]));
	LinkedPoint* pts = (LinkedPoint*)malloc((img_rows*img_cols+1)*sizeof(pts[0]));
	int ER_no = get_ERs(img_data, img_rows, img_cols, 0/*2:see debug msg*/, ERs, pts);

	// assign some global variables
	G_td.img = &img;
	G_td.ERs = ERs;
	G_td.ER_no = ER_no;
	G_td.pts = pts;

	// prepare for getting ER candidates
	G_td.featraw = (featraw_t *)malloc(ER_no*sizeof(featraw_t));
	memset(G_td.featraw, 0, ER_no*sizeof(featraw_t));

	// get ER candidates
	get_ER_candidates();

	printf("main_sample_1 test is good\n");
	char ch;
	scanf("%c", &ch);

	free(ERs);
	free(pts);
	
	return 0;
}

int main(void)
{

	main_sample_2();
#if 0
		// Boost parameters
		CvBoostParams bstparams;
		bstparams.boost_type = CvBoost::REAL;
		bstparams.weak_count = 100;
		bstparams.weight_trim_rate = 0.95;
		bstparams.split_criteria = CvBoost::DEFAULT;

		// Run the training
		CvBoost *boost = new CvBoost;
		boost.train(featureVectorSamples, 
					CV_ROW_SAMPLE, 
					classLabelResponses, 
					0, 0, var_type, 
					0, bstparams);
#endif

#if 0
	char input[128];
	FILE *fin;
	fin = fopen(, "r");
	fscanf(fin, "%s", input);
	int img_no = atoi(input);

	char path_prefix[10] = "../../../";
	for (int i=1; i<=img_no; i++)
	{
		fscanf(fin, "%s", input);
		IplImage *img=cvLoadImage(strcat(path_prefix, input));

		cvtColor(img, grayImg, CV_BGR2GRAY);
		int a = 0;

	}

	if (file) {
		while (fscanf(file, "%s", str)!=EOF)
			printf("%s",str);
		fclose(file);
	}

	IplImage *img=cvLoadImage("../../../../../../Dataset/MSRA-TD500/test/IMG_0059.JPG");
	cvNamedWindow("a");
	cvShowImage("a", img);
	cvWaitKey(0);
	//CvMat img = imread("aPICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE);

	//int* out = (int*)malloc( (img.rows*img.cols*3)*sizeof(int) );
	//int* pxl = (int*)malloc( (img.rows*img.cols)*sizeof(int) );
	//get_ERs(&img, out, pxl, 0);

	//struct aa a;
#endif
	//fclose(fin);

	return 0;
}