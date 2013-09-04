
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

#define printff if(0)printf //printf

/* global variable */
G_textdetect_t G_td;

/* plot utility function */
void plot_ER(ER_t *T)
{
	u8 *img_data = (u8 *)malloc(G_td.img->rows*(*G_td.img->step.p)*sizeof(u8)); 
	memset(img_data, 0, G_td.img->rows*(*G_td.img->step.p)*sizeof(u8));
	LinkedPoint *cur = T->ER_head;
	for (int j=0; j<T->ER_size; j++) {
		img_data[cur->pt.y*(*G_td.img->step.p)+cur->pt.x] = 255;
		cur = cur->next;
	}
	CvSize size = {(*G_td.img->step.p), G_td.img->rows};
	IplImage *dst = cvCreateImage(size, 8, 1);
	dst->imageData = (char *)img_data;

	cvNamedWindow("a");
	cvShowImage("a", dst);
	cvWaitKey(0);

	free(img_data);
}

/* save utility function */
void save_ER(ER_t *T, int idx, FILE *f)
{
	u8 *img_data = (u8 *)malloc(G_td.img->rows*(*G_td.img->step.p)*sizeof(u8)); 
	memset(img_data, 0, G_td.img->rows*(*G_td.img->step.p)*sizeof(u8));
	LinkedPoint *cur = T->ER_head;
	for (int j=0; j<T->ER_size; j++) {
		img_data[cur->pt.y*(*G_td.img->step.p)+cur->pt.x] = 255;
		cur = cur->next;
	}
	CvSize size = {(*G_td.img->step.p), G_td.img->rows};
	IplImage *dst = cvCreateImage(size, 8, 1);
	dst->imageData = (char *)img_data;

	//char filepath[100];
	//sprintf(filepath,"../../../../../../../LargeFiles/c_impl/0412/[%03d]/%05d.jpg", G_td.img_id, idx);
	//cvSaveImage(filepath, dst);
	char filepath[100], filename[64];
	int pathlen = strlen(G_td.output_path);
	//sprintf(filename, "%05d.jpg", idx);
#if 1
	fprintf(f, "%d	%d	%d	%d	%c\n", T->l, T->t, T->r-T->l+1, T->b-T->t+1, G_td.channel);
#else
	sprintf(filename, "[%d][%c][%04d-%04d-%04d-%04d][%d].jpg", 
		G_td.img_id, G_td.channel, T->l, T->t, T->r-T->l+1, T->b-T->t+1, G_td.r.text_is_darker);
	strcpy(filepath, G_td.output_path);
	strcpy(&filepath[pathlen], filename);
	cvSaveImage(filepath, dst);
#endif
	free(img_data);
}

/*
procedure TREE-ACCUMULATION(T)
2: if nchildren[T] >= 2 then
3:		C = empty;
4:		for each c in children[T] do
5:			C = C union TREE-ACCUMULATION(c)
6:		end for
7:		if var[T] <= min-var[C] then
8:			discard-children(T)
9:			return T
10:		else
11:			return C
12:		end if
13: else // nchildren[T] = 0
14:		return T
15: end if
end procedure
*/

/* Use size variace with aspect ratio penalty */
bool tree_accumulation_algo1(ER_t *T, int C_no, ER_un_t *C)
{
	//if var[T] <= min-var[C] then
	// return true: T is better
	// return false: c is better
	if (C_no == 0)
		return true;
	ER_un_t *cur = C;
	T->ar = (T->r - T->l + 1) * 1.0 / (T->b - T->t + 1);

	// find min_svar_wp_C (with penalty)
	double svar_wp_T = (T->to_parent) ? ((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size) : 0;
	if (T->ar > G_td.r.max_ar)
		svar_wp_T = svar_wp_T - (G_td.r.large_ar_pnty_coef) * (T->ar - G_td.r.max_ar);
	else if (T->ar < G_td.r.min_ar)
		svar_wp_T = svar_wp_T - (G_td.r.small_ar_pnty_coef) * (G_td.r.max_ar - T->ar);
	double min_svar_wp_C = 10000000; cur = C;
	for (int i=0; i<C_no; i++, cur = cur->next) {
		cur->ER->ar = (cur->ER->r - cur->ER->l + 1) * 1.0 / (cur->ER->b - cur->ER->t + 1);
		double svar_wp_c = (cur->ER->to_parent->ER_size - cur->ER->ER_size)*1.0 / cur->ER->ER_size;
		if (cur->ER->ar > G_td.r.max_ar)
			svar_wp_c = svar_wp_c - (G_td.r.large_ar_pnty_coef) * (cur->ER->ar - G_td.r.max_ar);
		else if (cur->ER->ar < G_td.r.min_ar)
			svar_wp_c = svar_wp_c - (G_td.r.small_ar_pnty_coef) * (G_td.r.max_ar - cur->ER->ar);

		if (svar_wp_c < min_svar_wp_C) min_svar_wp_C = svar_wp_c;
	}

	// compare var_T with min_var_C
	if (min_svar_wp_C < svar_wp_T)
		return false;
	else
		return true;
}

/* Use posterior prob first then size variace */
bool tree_accumulation_algo2(ER_t *T, int C_no, ER_un_t *C)
{
	//if var[T] <= min-var[C] then
	// return true: T is better
	// return false: c is better
	if (C_no == 0)
		return true;
	ER_un_t *cur = C;

	// find min_svar_C
	double svar_T = (T->to_parent) ? ((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size) : 0;
	double min_svar_C = 10000000; cur = C;
	for (int i=0; i<C_no; i++, cur = cur->next) {
		double svar_c = (cur->ER->to_parent->ER_size - cur->ER->ER_size)*1.0 / cur->ER->ER_size;
		if (svar_c < min_svar_C) min_svar_C = svar_c;
	}

	// find max_post_C
	double post_T = T->postp;//calc_postp_by_feat_ar_cp_nh(T);
	double max_post_C = 0; cur = C;
	for (int i=0; i<C_no; i++, cur = cur->next) {
		double post_c = cur->ER->postp;
		if (post_c >= max_post_C) max_post_C = post_c;
	}

	// compare var_T with min_var_C
	if (post_T >= max_post_C) {
		if (min_svar_C < svar_T) 
			return false;
		else
			return true;
	} else
		return false;
}

/* Size variace with aspect ratio penalty */
bool tree_accumulation_algo3(ER_t *T, int C_no, ER_un_t *C)
{
	//if var[T] <= min-var[C] then
	// return true: T is better
	// return false: c is better
	if (C_no == 0)
		return true;
	ER_un_t *cur = C;
	T->ar = (T->r - T->l + 1) * 1.0 / (T->b - T->t + 1);

	// find min_svar_wp_C (with penalty)
	double svar_wp_T = (T->to_parent) ? ((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size) : 0;
	if (T->ar > G_td.r.max_ar)
		svar_wp_T = svar_wp_T - (G_td.r.large_ar_pnty_coef) * (T->ar - G_td.r.max_ar);
	else if (T->ar < G_td.r.min_ar)
		svar_wp_T = svar_wp_T - (G_td.r.small_ar_pnty_coef) * (G_td.r.max_ar - T->ar);
	double min_svar_wp_C = 10000000; cur = C;
	for (int i=0; i<C_no; i++, cur = cur->next) {
		cur->ER->ar = (cur->ER->r - cur->ER->l + 1) * 1.0 / (cur->ER->b - cur->ER->t + 1);
		double svar_wp_c = (cur->ER->to_parent->ER_size - cur->ER->ER_size)*1.0 / cur->ER->ER_size;
		if (cur->ER->ar > G_td.r.max_ar)
			svar_wp_c = svar_wp_c - (G_td.r.large_ar_pnty_coef) * (cur->ER->ar - G_td.r.max_ar);
		else if (cur->ER->ar < G_td.r.min_ar)
			svar_wp_c = svar_wp_c - (G_td.r.small_ar_pnty_coef) * (G_td.r.max_ar - cur->ER->ar);

		if (svar_wp_c < min_svar_wp_C) min_svar_wp_C = svar_wp_c;
	}

	if (min_svar_wp_C > svar_wp_T)
		return false;
	else
		return true;
}

ER_un_t *tree_accumulation(ER_t *T, int *C_no_this)
{
	if (T->ER_size < G_td.r.min_size)
		return NULL;
	if (T->ER_noChild == 0) {
		// has no child
		/// return single union node T
		G_td.ER_un[T->ER_id].ER = T;
		G_td.ER_un[T->ER_id].prev = NULL;
		G_td.ER_un[T->ER_id].next = NULL;
		(*C_no_this)++;
		return &G_td.ER_un[T->ER_id];

	} else if (T->ER_noChild >= 2) {
		// has more than two children
		int C_no = 0;
		ER_un_t *C = NULL, *C_cur = NULL, *C_ret = NULL;
		ER_t *c = T->to_firstChild;
		while (c) {
			/// obtain accumulated union C_ret head 
			C_ret = tree_accumulation(c, &C_no);
			/// union by linking current union C tail with accumlated union C_ret head
			if (C_ret != NULL) {
				if (C == NULL) {
					C = C_ret;
				} else {
					C_cur->next = C_ret;
					C_ret->prev = C_cur;
				}
				ER_un_t *cur = C_ret;
				while (cur->next)
					cur = cur->next;
				C_cur = cur;
			}
			c = c->to_nextSibling;
		}
		int T_is_better = 0;
		if ((T->ER_size <= G_td.r.max_size) && (G_td.ta_algo(T, C_no, C))) {
			T_is_better = 1;
		}
		if (T_is_better) {
			// discard-children(T)
			T->ER_firstChild = -1;
			T->to_firstChild = NULL;

			// return single union node T
			G_td.ER_un[T->ER_id].ER = T;
			G_td.ER_un[T->ER_id].prev = NULL;
			G_td.ER_un[T->ER_id].next = NULL;
			(*C_no_this)++;

			return &G_td.ER_un[T->ER_id];
		} else {
			// return current union head node C
			(*C_no_this) += C_no;
			return C;
		}
	} else {
		// has only one child
		// after linear reduction, there should be no such case
		assert(0);
	}
}

/*
procedure LINEAR-REDUCTION(T)
2: if nchildren[T] = 0 then
3:		return T
4: else if nchildren[T] = 1 then
5:		c = LINEAR-REDUCTION(child[T])
6:		if var[T] <= var[c] then
7:			link-children(T, children[c])
8:			return T
9:		else
10:			return c
11:		end if
12:else // nchildren[T] = 2
13:		for each c 2 children[T] do
14:			link-children(T, LINEAR-REDUCTION(c))
15:		end for
16:		return T
17:end if
end procedure
*/

bool linear_reduction_algo(ER_t *T, ER_t *c)
{
	// Before:        --> "T" --> "c" --> "c's child"
	// After(True):   --> "T" ----------> "c's child"
	// After(False):  ----------> "c" --> "c's child"
	double svar_T = (T->to_parent) ? ((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size) : 0;
	double svar_c = (T->ER_size - c->ER_size) * 1.0 / c->ER_size;
	if (svar_T <= svar_c) {
		// T's variance <= c's variance
		return true;
	} else {
		// T's variance > c's variance
		return false;
	}
}

ER_t *linear_reduction(ER_t *T)
{
	// mark invalid for extream size ER
	if ((T->ER_size < G_td.r.min_size) || (T->ER_size > G_td.r.max_size)) {
		G_td.ER_no_array[T->ER_id] = 0;
	}

	if (T->ER_noChild == 0) {
		// has no child
		return T;
	} else if (T->ER_noChild == 1) {
		// has only one child
		G_td.ER_no_rest--;
		ER_t *c = linear_reduction(T->to_firstChild);
		if (G_td.lr_algo(T,c)){
			// remove c, link T to its new child, "c's child"
			T->ER_noChild = c->ER_noChild;
			T->ER_firstChild = c->ER_firstChild;
			T->to_firstChild = c->to_firstChild;
			ER_t *t = T->to_firstChild;
			while(t) {
				t->ER_parent = T->ER_id;
				t->to_parent = T;
				t = t->to_nextSibling;
			}
			G_td.ER_no_array[c->ER_id] = 0;
			return T; // T is better
		} else {
			G_td.ER_no_array[T->ER_id] = 0;
			return c; // c is better
		}
	} else {
		// has more than two children
		ER_t *c = T->to_firstChild;
		int childNo = 0;
		while (c) {
			ER_t *d = linear_reduction(c);
			// link T to its new child "d"
			childNo++;
			if ((T->ER_firstChild!=d->ER_id) && (c->ER_id!=d->ER_id)) {
				if (childNo==1) {
					if (T->to_firstChild->to_nextSibling) {
						T->to_firstChild->to_nextSibling->to_prevSibling = d;
						T->to_firstChild->to_nextSibling->ER_prevSibling = d->ER_id;
						d->to_nextSibling = T->to_firstChild->to_nextSibling;
						d->ER_nextSibling = T->to_firstChild->ER_nextSibling;
					}
					T->ER_firstChild = d->ER_id;
					T->to_firstChild = d;
				} else {
					d->ER_nextSibling = c->ER_nextSibling;
					d->to_nextSibling = c->to_nextSibling;
					if (d->to_nextSibling) {
						d->to_nextSibling->to_prevSibling = d;
						d->to_nextSibling->ER_prevSibling = d->ER_id;
					}
					d->ER_prevSibling = c->ER_prevSibling;
					d->to_prevSibling = c->to_prevSibling;
					if (d->to_prevSibling) {
						d->to_prevSibling->to_nextSibling = d;
						d->to_prevSibling->ER_nextSibling = d->ER_id;
					}
				}
				d->ER_parent = T->ER_id;
				d->to_parent = T;
			}
			// next T's child
			c = c->to_nextSibling;
		}
		return T;
	}
}

void get_ER_candidates(void)
{
	/* Hook up boost classifier */
	CvBoost boost;
	boost.load("./boost_4fv.xml", "boost");
	G_td.boost = (CvBoost *)&boost;

	/* Linear reduction */
	G_td.ER_no_rest = G_td.ER_no;
	printff("[original] ER rest : %d\n", G_td.ER_no_rest);
	G_td.lr_algo = linear_reduction_algo;
	ER_t *root = &G_td.ERs[G_td.ER_no-1];
	root = linear_reduction(root);

	if (G_td.r.tree_accum_algo == 2) {
		/* Calc postp at a time */
		int m = 0;
		CvMat* featVector = cvCreateMat(1, 4, CV_32FC1);
		for (int i=0; i<G_td.ER_no; i++) {
			if (G_td.ER_no_array[i]) {
				m++;
				/* prepare features (1~3) : aspect ratio, compactness, no of holes */
				float ar = (G_td.ERs[i].r - G_td.ERs[i].l + 1) * 1.0 / (G_td.ERs[i].b - G_td.ERs[i].t + 1);  
				float cp = sqrt((double)G_td.ERs[i].ER_size) * 1.0 / G_td.ERs[i].p;
				float nh = 1 - G_td.ERs[i].eu;
				/* prepare feature (4) : median of horizontal crossing */
				int h = G_td.ERs[i].b - G_td.ERs[i].t + 1;
				int w = G_td.ERs[i].r - G_td.ERs[i].l + 1;
				float h1 = floor(h*1.0/6);
				float h2 = floor(h*3.0/6);
				float h3 = floor(h*5.0/6);
				int hc1 = 0, hc2 = 0, hc3 = 0;
				LinkedPoint *cur = G_td.ERs[i].ER_head;
				memset(G_td.hc1, 0, G_td.img->cols*sizeof(u8));
				memset(G_td.hc2, 0, G_td.img->cols*sizeof(u8));
				memset(G_td.hc3, 0, G_td.img->cols*sizeof(u8));
				for (int k=0; k<G_td.ERs[i].ER_size; k++, cur=cur->next) {
					if (cur->pt.y == h1)
						G_td.hc1[cur->pt.x] = 1;
					if (cur->pt.y == h2)
						G_td.hc2[cur->pt.x] = 1;
					if (cur->pt.y == h3)
						G_td.hc3[cur->pt.x] = 1;
				}
				for (int k=0; k<w-1; k++) {
					if (G_td.hc1[k] + G_td.hc1[k+1] == 1) hc1++;
					if (G_td.hc2[k] + G_td.hc2[k+1] == 1) hc2++;
					if (G_td.hc3[k] + G_td.hc3[k+1] == 1) hc3++;
				}
				int hc[3];
				hc[0] = hc1 + (G_td.hc1[0] + G_td.hc1[w-1]);
				hc[1] = hc2 + (G_td.hc2[0] + G_td.hc2[w-1]);
				hc[2] = hc3 + (G_td.hc3[0] + G_td.hc3[w-1]);
				for (int k=2; k>0; k--) {
					for (int j=2; j>0; j--) {
						if (hc[j]<hc[j-1]) {
							int tmp = hc[j-1];
							hc[j-1] = hc[j];
							hc[j] = tmp;
						}
					}
				}
				// calc posterior probability
				featVector->data.fl[0] = ar;
				featVector->data.fl[1] = cp;
				featVector->data.fl[2] = nh;
				featVector->data.fl[3] = hc[1]; // median
				float score = boost.predict(featVector, 0, 0, CV_WHOLE_SEQ, false, true);
				G_td.ERs[i].label = (score>=0) ? 1 : -1;
				G_td.ERs[i].postp = 1.0/(1+exp(-2.0*abs(score)));
				G_td.ERs[i].ar = ar;
			}
		}
	}
#if 0
	for (int i=0; i<G_td.ER_no; i++) {
		if (G_td.ER_no_array[i]) {
			if (G_td.ERs[i].ER_id > 10000)
				continue;
			save_ER(&G_td.ERs[i], G_td.ERs[i].ER_id);
		}
	}
#endif

	/* Tree accumulation */
	int no_union = 0;
	printff("[li_reduc] ER rest : %d\n", G_td.ER_no_rest);
	if (G_td.r.tree_accum_algo == 1) G_td.ta_algo = tree_accumulation_algo1;
	if (G_td.r.tree_accum_algo == 2) G_td.ta_algo = tree_accumulation_algo2;
	if (G_td.r.tree_accum_algo == 3) G_td.ta_algo = tree_accumulation_algo3;
	ER_un_t *C_union = tree_accumulation(root, &no_union);
	printff("[tr_accum] ER rest : %d\n", no_union);

#if 1
	
	char fn[64];
	sprintf(fn, "%s%03d.txt", G_td.output_path, G_td.img_id);
	FILE *f = fopen(fn, "a");
	ER_un_t *cur = C_union;
	for (int i=0; i<no_union; i++, cur = cur->next) {
		save_ER(cur->ER, cur->ER->ER_id, f);
	}
	fclose(f);
#endif

}

#if 0
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
			double percent = 100;
			//IplImage *src = cvLoadImage((path_prefix + path_img).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			//IplImage *src = cvLoadImage("../../../../../Dataset/ICDAR_Robust_Reading/SceneTrialTest/ryoungt_05.08.2002/PICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE);
			//IplImage *src = cvLoadImage("PICT0034_Syn.JPG", CV_LOAD_IMAGE_GRAYSCALE); //BUS(syn)
			//IplImage *src = cvLoadImage("PICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE); //BUS
			//IplImage *src = cvLoadImage("dPICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE); //H319810
			//IplImage *src = cvLoadImage("test02.JPG", CV_LOAD_IMAGE_GRAYSCALE); //Citizen
			IplImage *src = cvLoadImage("dpCT0001.JPG", CV_LOAD_IMAGE_GRAYSCALE); // YAMAHA
			IplImage *dst = cvCreateImage(cvSize((int)((src->width*percent)/100), (int)((src->height*percent)/100) ), src->depth, src->nChannels);
			//IplImage *dst = cvCreateImage(cvSize(200, 200), src->depth, src->nChannels);
			cvResize(src, dst, CV_INTER_LINEAR);
			img = dst;
			//cvNamedWindow("a");
			//cvShowImage("a", dst);
			//cvWaitKey(0);
			char filepath[100];
			sprintf(filepath,"../../../../../../../LargeFiles/c_impl/0412/[%03d]/_originl.jpg", ii);
			cvSaveImage(filepath, dst);
		}
		//printf("Load image (size %d x %d) ...\n", img.cols, img.rows);
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock();

		char out[100];
		sprintf(out,"../../../../../../../LargeFiles/c_impl/0412/[%03d]/", ii);
		text_detect(&img, 1, out, 1);

		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock();

		printf("[%d] main_sample_2 test is good\n", ii);
		char ch;
		scanf("%c", &ch);
	}
}

int main_sample_3(void)
{
	// Train boost classifier
	Mat featureVectorSample;
	Mat classLabelResponse;
	string demoFile = "../../../../../Codes/_output_files/Feature_vectors/fv.yml";
	FileStorage fsDemo(demoFile, FileStorage::READ);
	fsDemo["fv_save"] >> featureVectorSample;
	fsDemo["lb_save"] >> classLabelResponse;
	int var_count = featureVectorSample.cols;
	int nsamples_all = featureVectorSample.rows;
	CvMat featureVectorSamples = featureVectorSample;
	CvMat classLabelResponses = classLabelResponse;

	CvMat* var_type = cvCreateMat(var_count + 1, 1, CV_8U);
	cvSet(var_type, cvScalarAll(CV_VAR_ORDERED)); // Inits all to 0 like the code below
	var_type->data.ptr[var_count] = CV_VAR_CATEGORICAL;

	CvBoost boost;
	//boost.train(&featureVectorSamples, CV_ROW_SAMPLE, &classLabelResponses, 0, 0, var_type, 0, CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0));
	//boost.save("./boost_4fv.xml", "boost");

	// Test boost classifier
	boost.load("./boost_4fv.xml", "boost");
	const float awesome_data[3] = {0.111, 0.111, 0.111}; 
	CvMat testSamples = cv::Mat(1,3,CV_32FC1, (void*)awesome_data);

	float ans1, ans2;
	//cvGetRows(&featureVectorSamples, &testSamples, 1, 2);
	ans1 = boost.predict(&testSamples, 0, 0, CV_WHOLE_SEQ, false, true);
	ans2 = boost.predict(&testSamples);

	cvGetRows(&featureVectorSamples, &testSamples, 3000, 3001);
	ans1 = boost.predict(&testSamples, 0, 0, CV_WHOLE_SEQ, false, true);
	ans2 = boost.predict(&testSamples);

	return 0;
}
#endif

void text_detect(Mat *img, int text_is_darker, char *output_path, int algo, int cur_img_id, char cur_channel)
{
	G_td.img = img;                                // input image
	G_td.channel = cur_channel;                    // current channel name
	G_td.img_id = cur_img_id;                      // current image id
	G_td.output_path = output_path;                // output folder path
	G_td.r.text_is_darker = text_is_darker;        // text is darker than background or not
	G_td.r.tree_accum_algo = algo;                 // 1,2,3
	
	/* The following are default value, can be changed here */
	G_td.r.min_reg2img_ratio = 0.001;              // min region to img ratio
	G_td.r.max_reg2img_ratio = 0.25;               // max region to img ratio
	if (G_td.r.tree_accum_algo != 2) {             // extra parameter is needed for algo 2
		G_td.r.min_ar = 0.7;
		G_td.r.max_ar =	1.2;
		G_td.r.small_ar_pnty_coef = 0.08;
		G_td.r.large_ar_pnty_coef = 0.03;
	}

	// get ERs
	ER_t *ERs = (ER_t *)malloc(img->rows*img->cols*sizeof(ERs[0]));
	G_td.ERs = ERs;
	LinkedPoint *pts = (LinkedPoint*)malloc((img->rows*img->cols+1)*sizeof(pts[0]));
	int ER_no = get_ERs(img->data, img->rows, img->cols, *(img->step.buf), !G_td.r.text_is_darker/*2:see debug msg*/, ERs, pts);

	// assign some global variables
	G_td.ER_no = ER_no;
	G_td.pts = pts;

	// prepare for getting ER candidates
	G_td.featraw = (featraw_t *)malloc(ER_no*sizeof(featraw_t));
	memset(G_td.featraw, 0, ER_no*sizeof(featraw_t));
	G_td.ER_no_array = (u8 *)malloc(ER_no*sizeof(u8));
	memset(G_td.ER_no_array, 1 , ER_no*sizeof(u8));
	G_td.ER_un = (ER_un_t *)malloc(ER_no*sizeof(ER_un_t));
	memset(G_td.ER_un, 0 , ER_no*sizeof(ER_un_t));
	G_td.r.max_size = G_td.r.max_reg2img_ratio * img->cols * img->rows;//from ratio to real size
	G_td.r.min_size = G_td.r.min_reg2img_ratio * img->cols * img->rows;//from ratio to real size
	G_td.r.min_size = MAX(G_td.r.min_size, 64);
	G_td.hc1 = (u8 *)malloc(img->cols*sizeof(u8));
	G_td.hc2 = (u8 *)malloc(img->cols*sizeof(u8));
	G_td.hc3 = (u8 *)malloc(img->cols*sizeof(u8));

	// get ER candidates
	get_ER_candidates();
	free(ERs);
	free(pts);
	for (int i=0; i<ER_no; i++) {
		if (G_td.featraw[i].HC_buf)
			free(G_td.featraw[i].HC_buf);
	}
	free(G_td.featraw);
}

void text_detect(Mat *img, int text_is_darker, char *output_path, int algo)
{
	int cur_img_id = 0;
	char cur_channel = ' ';
	text_detect(img, text_is_darker, output_path, algo, cur_img_id, cur_channel);
}

int main(void)
{

#if 0
	//Sign on the street
	Mat img = imread("PICT0017.JPG", CV_LOAD_IMAGE_GRAYSCALE); 
	char out[100] = "../../test/PICT0017/"; 
	text_detect(&img, 0, out, 1);
		//BUS
	Mat img = imread("PICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE); 
	char out[100] = "../../test/PICT0034/";
	text_detect(&img, 1, out, 3);
#else
	

	//cvShowImage("src", src);
	//cvShowImage("dst", dst);
	//cvShowImage("y", &(IplImage)yy);
	//cvShowImage("u", &(IplImage)uu);
	//cvShowImage("v", &(IplImage)vv);
	//cvWaitKey(0);
	
	int algo = 1;
	int img_id = 0;

	char in[100] =  "../../../../../Dataset/ICDAR_2013/SceneTest/";
	char out[100] = "../../../../../../../LargeFiles/ICDAR_2013/";
	for (int img_id = 4; img_id<=233; img_id++) {
		
		char fn[128];
		sprintf(fn, "%simg_%d.jpg", in, img_id);
		IplImage* src = cvLoadImage(fn, CV_LOAD_IMAGE_COLOR);
		IplImage* dst = cvCloneImage(src);
		cvCvtColor(src, dst, CV_RGB2YUV);
		CvSize s = cvGetSize(dst);
		IplImage *y = cvCreateImage(s,IPL_DEPTH_8U,CV_8UC1),
				 *u = cvCreateImage(s,IPL_DEPTH_8U,CV_8UC1),
				 *v = cvCreateImage(s,IPL_DEPTH_8U,CV_8UC1); 
		cvSplit(dst, y, u, v, NULL);
		Mat yy = Mat(y,0);
		Mat uu = Mat(u,0);
		Mat vv = Mat(v,0);

		text_detect(&yy, 0, out, algo, img_id, 'y');
		text_detect(&yy, 1, out, algo, img_id, 'y');
		text_detect(&uu, 0, out, algo, img_id, 'u');
		text_detect(&uu, 1, out, algo, img_id, 'u');
		text_detect(&vv, 0, out, algo, img_id, 'v');
		text_detect(&vv, 1, out, algo, img_id, 'v');
	}

#endif

	return 0;
}
