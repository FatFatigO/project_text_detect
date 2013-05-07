
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
void save_ER(ER_t *T, int idx)
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

	char filepath[100];
	sprintf(filepath,"../../../../../../../LargeFiles/c_impl/0412/[%03d]/%05d.jpg", G_td.img_id, idx);
	cvSaveImage(filepath, dst);

	free(img_data);
}



void calc_incremental_BB(int feat_id_to)
{
	// init "from" param
	p4_t *feat_fr_BB = NULL;

	// init "to" param
	featraw_t *feat_to = &G_td.featraw[feat_id_to];
	int pt_to_start_idx = G_td.ERs[feat_id_to].ER_head->pt_order;
	int pt_to_size = G_td.ERs[feat_id_to].ER_size;

	// init "pt" param
	p8_t pt;
	pt.val[0] = (u32)G_td.pts;
	pt.val[1] = G_td.img->rows;
	pt.val[2] = G_td.img->cols;
	pt.val[3] = 0;
	pt.val[4] = NULL;
	pt.val[5] = NULL;
	pt.val[6] = pt_to_start_idx;
	pt.val[7] = pt_to_size;

	// bounding box
	get_BoundingBox(IN &pt, IN feat_fr_BB, OUT &feat_to->BB);
}


void calc_incremental_feature_multi(int no_fr, int *feat_id_fr, int feat_id_to)
{
	return; ///////////////////////////

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
	//get_BoundingBox(IN &pt, IN feat_fr_BB, OUT &feat_to->BB);

	// perimeter
	//get_Perimeter(IN &pt, IN feat_fr_PR, OUT &feat_to->PR);

	// euler no
	//get_EulerNo(IN &pt, IN feat_fr_EN, OUT &feat_to->EN);

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

bool linear_reduction_algo_1(ER_t *T, ER_t *c)
{
	// return true: T is better
	// return false: c is better
	if (c->ER_size < G_td.r.min_size) {
		return true;
	} else if ((((c->r - c->l + 1)*1.0 / G_td.img->cols) < G_td.r.min_w_reg2img_ratio) ||
			   (((c->b - c->t + 1)*1.0 / G_td.img->rows) < G_td.r.min_h_reg2img_ratio)) {
		return true;	   
	} else if ((((c->r - c->l + 1)*1.0 / G_td.img->cols) > G_td.r.max_w_reg2img_ratio) ||
			   (((c->b - c->t + 1)*1.0 / G_td.img->rows) > G_td.r.max_h_reg2img_ratio)) {
		return true;
	} else if (((T->to_parent->ER_size - T->ER_size)*1.0 / T->ER_size) <
		       ((c->to_parent->ER_size - c->ER_size)*1.0 / c->ER_size))
		return true;
	else
		return false;
}
bool tree_accumulation_algo_1(ER_t *T, int C_no, ER_un_t *C)
{
	//if var[T] <= min-var[C] then
	// return true: T is better
	// return false: c is better
	if (T->ER_id == 355)
		T = T;
	double min_var_C = 10000000;
	double min_pvar_C = 10000000;
	ER_un_t *cur = C;
	for (int i=0; i<C_no; i++) {
		double var_c = (cur->ER->to_parent->ER_size - cur->ER->ER_size)*1.0 / cur->ER->ER_size;
		double pvar_c = (cur->ER->to_parent->p - cur->ER->p) * 1.0 / cur->ER->p;
		if (var_c==0)
			T = T;
		if (var_c < min_var_C)
			min_var_C = var_c;
		if (pvar_c < min_pvar_C)
			min_pvar_C = pvar_c;
		cur = cur->next;
	}

	double pvar_T = (T->to_parent) ? ((T->to_parent->p - T->p) * 1.0 / T->p) : 0;

	if ((T->to_parent) && (((T->to_parent->ER_size - T->ER_size)*1.0 / T->ER_size) <= min_var_C)) {
		save_ER(T, 999999);
		cur = C;
		for (int i=0; i<C_no; i++) {
			save_ER(cur->ER, i);
			cur = cur->next;
		}
		
		return true;
	} else
		return false;
}

bool linear_reduction_algo(ER_t *T, ER_t *c)
{
	// Before:        --> "T" --> "c" --> "c's child"
	// After(True):   --> "T" ----------> "c's child"
	// After(False):  ----------> "c" --> "c's child"
	double pvar_T = (T->to_parent) ? ((T->to_parent->p - T->p) * 1.0 / T->p) : 0;
	double pvar_c = (T->p - c->p) * 1.0 / c->p;
	double svar_T = (T->to_parent) ? ((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size) : 0;
	double svar_c = (T->ER_size - c->ER_size) * 1.0 / c->ER_size;

	double lumbda_1 = 1;
	double lumbda_2 = 1;
	double var_T = (pvar_T > 0) ? (svar_T + lumbda_1 * pvar_T) : (svar_T + lumbda_2 * pvar_T);
	double var_c = (pvar_c > 0) ? (svar_c + lumbda_1 * pvar_c) : (svar_c + lumbda_2 * pvar_c);

	if (0){//(T->ER_size > 100) {
		cout << "T: size var=" << svar_T << endl;
		cout << "T: peri var=" << pvar_T << endl;
		plot_ER(T);
		cout << "c: size var=" << svar_c << endl;
		cout << "c: peri var=" << pvar_c << endl;
		plot_ER(c);
		cout << "T: var=" << var_T << endl;
		cout << "c: var=" << var_c << endl;
	}

	if (svar_T <= svar_c) {
		// T's variance <= c's variance
		//cout << "T is better!" << endl;
		//int a;
		//cin >> a;
		return true;
	} else {
		// T's variance > c's variance
		//cout << "c is better!" << endl;
		//int a;
		//cin >> a;
		return false;
	}
}
bool linear_reduction_algo2(ER_t *T, ER_t *c)
{
	// "T" --> "c" --> "c's child"
	if ((T->to_parent) &&
		((T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size <=
	     (T->ER_size - c->ER_size) * 1.0 / c->ER_size)) {
		// T's variance <= c's variance
		if (0) {//(T->ER_size > 100) {
			if (T->p > 9999)
				T = T;
			cout << "T: size var=" << (T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size << endl;
			cout << "T: peri var=" << (T->to_parent->p - T->p) * 1.0 / T->p << " " << T->p << "->" << T->to_parent->p << endl;
			plot_ER(T);
			cout << "c: size var=" << (T->ER_size - c->ER_size) * 1.0 / c->ER_size << endl;
			cout << "c: peri var=" << (T->p - c->p) * 1.0 / c->p << " " << c->p << "->" << T->p << endl;
			cout << "T is better!" << endl;
			plot_ER(c);
		}

		return true;
	} else {
		// T's variance > c's variance
		if (0) {//(T->ER_size > 100) {
			cout << "T: size var=" << (T->to_parent->ER_size - T->ER_size) * 1.0 / T->ER_size << endl;
			cout << "T: peri var=" << (T->to_parent->p - T->p) * 1.0 / T->p << " " << T->p << "->" << T->to_parent->p << endl;
			plot_ER(T);
			cout << "c: size var=" << (T->ER_size - c->ER_size) * 1.0 / c->ER_size << endl;
			cout << "c: peri var=" << (T->p - c->p) * 1.0 / c->p << " " << c->p << "->" << T->p << endl;
			cout << "c is better!" << endl;
			plot_ER(c);
		}
		return false;
	}
}
bool tree_accumulation_algo(ER_t *T, int C_no, ER_t **C)
{
	//if var[T] <= min-var[C] then
	return false;//true;
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

ER_un_t *tree_accumulation(ER_t *T, int *C_no_this)
{
	if (T->ER_id == 86)
		T = T;
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
			if (C_no == 5)
				C_no = C_no;
			if (c == T->to_firstChild) {
				C = C_ret;
			} else {
				C_cur->next = C_ret;
				C_ret->prev = C_cur;
			}
			ER_un_t *cur = C_ret;
			while (cur->next)
				cur = cur->next;
			C_cur = cur;
			c = c->to_nextSibling;
		}
		if (T->ER_id == 116)
			T = T;
		if (G_td.ta_algo(T, C_no, C)) {
			//discard-children(T)
			T->ER_firstChild = -1;
			T->to_firstChild = NULL;
			//return T

			/// return single union node T
			G_td.ER_un[T->ER_id].ER = T;
			G_td.ER_un[T->ER_id].prev = NULL;
			G_td.ER_un[T->ER_id].next = NULL;
			(*C_no_this)++;

			return &G_td.ER_un[T->ER_id];
		} else {
			//return C

			/// return current union head node C
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

ER_t *linear_reduction(ER_t *T)
{
	int a = T->ER_id;
#if 0
	if (T->ER_size > 100) {
		cout << a << endl;
		plot_ER(T);
		ER_t *cur = T->to_firstChild;
		/*
		while(cur) {
			if (cur->ER_size > 100)
				plot_ER(cur);
			cur = cur->to_nextSibling;
		}*/
	}
#endif
	if (T->ER_noChild == 0) {
		// has no child
		printff("[%d] has no child, return %d\n",a,T->ER_id);

		/* ---------------------------- START ---------------------------------*/
		// <====== Feature extraction (in/crementally computed from Null => T)
		calc_incremental_feature_multi(0, NULL, T->ER_id);
		/* ----------------------------- END ----------------------------------*/
		if ((T->ER_size < G_td.r.min_size) && (!(T->to_nextSibling) && !(T->to_prevSibling)))
			T = T;
		return T;
	} else if (T->ER_noChild == 1) {
		// has only one child
		G_td.ER_no_rest--;
		printff("[%d] %d has one child\n",a,T->ER_id);
		printff("[%d] %d go linear reduction \n",a,T->to_firstChild->ER_id);
		ER_t *c = linear_reduction(T->to_firstChild);
		printff("[%d] %d returned from linear reduction \n",a,c->ER_id);
		if (G_td.lr_algo(T,c)){
			/* ---------------------------- START ---------------------------------*/
			// Feature extraction (incrementally computed from T->firstChild => T)
			int *fr_id = &T->to_firstChild->ER_id;
			calc_incremental_feature_multi(1, fr_id, T->ER_id);
			/* ----------------------------- END ----------------------------------*/

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
			printff("[%d] link %d to its new child %d\n",a,T->ER_id,c->ER_firstChild);
			printff("[%d] return %d \n",a,T->ER_id);
			G_td.ER_no_array[c->ER_id] = 0;
			return T; // T is better
		} else {
			printff("[%d] return %d \n",a,c->ER_id);
			G_td.ER_no_array[T->ER_id] = 0;
			return c; // c is better
		}
	} else {
		// has more than two children
		printff("[%d] %d has more than two child\n",a,T->ER_id);
		ER_t *c = T->to_firstChild;
		int childNo = 0;
		while (c) {
			printff("[%d] %d go linear reduction \n",a,c->ER_id);

			ER_t *d = linear_reduction(c);

			printff("[%d] %d returned from linear reduction \n",a,d->ER_id);
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
			printff("[%d] link %d to its new child %d\n",a,T->ER_id,d->ER_id);
			// next T's child
			c = c->to_nextSibling;
		}
		printff("[%d] return %d \n",a,T->ER_id);

		/* ---------------------------- START ---------------------------------*/
		// Feature extraction (incrementally computed from T->firstChild => T)
		int *fr_id = (int *)malloc(childNo*sizeof(int));
		c = T->to_firstChild;
		for (int i=0; i<T->ER_noChild; i++) {
			fr_id[i] = c->ER_id;
			c = c->to_nextSibling;
		}
		calc_incremental_feature_multi(T->ER_noChild, fr_id, T->ER_id);
		free(fr_id);
		/* ----------------------------- END ----------------------------------*/

		return T;
	}
}

void tree_remove_extreme_size_ER(ER_t *v)
{
	ER_t *d = v->to_firstChild;
	while (d) {
		tree_remove_extreme_size_ER(d);
		ER_t *c = d;
		d = d->to_nextSibling;

		//visit each ER;
		if ((c->ER_size < G_td.r.min_size) ||
			(((c->r - c->l + 1)*1.0 / G_td.img->cols) < G_td.r.min_w_reg2img_ratio) ||
			(((c->b - c->t + 1)*1.0 / G_td.img->rows) < G_td.r.min_h_reg2img_ratio)) {
			// remove node c
			// (1) parent related
			if (c->to_parent) {
				if (c->to_parent->ER_firstChild==c->ER_id) {
					if (c->to_nextSibling) {
						c->to_parent->ER_firstChild = c->ER_nextSibling;
						c->to_parent->to_firstChild = c->to_nextSibling;
					} else {
						c->to_parent->ER_firstChild = -1;
						c->to_parent->to_firstChild = NULL;
					}
				}
				c->ER_parent = -1;
				c->to_parent = NULL;
			}
			// (2) sibling related
			if (c->to_nextSibling) {
				c->to_nextSibling->ER_prevSibling = -1;
				c->to_nextSibling->to_prevSibling = NULL;
				c->ER_nextSibling = -1;
				c->to_nextSibling = NULL;
			}
			if (c->to_prevSibling) {
				c->to_prevSibling->ER_nextSibling = -1;
				c->to_prevSibling->to_nextSibling = NULL;
				c->ER_prevSibling = -1;
				c->to_prevSibling = NULL;
			}
			// (3) child related
			if (c->ER_noChild == 0) {
				// has no child
			} else if (c->ER_noChild == 1) {
				// has only one child
			} else {
				// has more than two children
			}
			c->ER_noChild = 0;
			// (4) invalid ER
			INVALID_ER(c);
			G_td.ER_no_rest--;
		}
	}
}


void get_ER_candidates(void)
{
	/* Remove extreme size */
	G_td.ER_no_rest = G_td.ER_no;
	printf("[original] ER rest : %d\n", G_td.ER_no_rest);
	//tree_remove_extreme_size_ER(&G_td.ERs[G_td.ER_no-1]);

	/* Linear reduction */
	//G_td.lr_algo = linear_reduction_algo_1;
	G_td.lr_algo = linear_reduction_algo;
	//printf("[rm_extrm] ER rest : %d\n", G_td.ER_no_rest);
	ER_t *root = &G_td.ERs[G_td.ER_no-1];
	root = linear_reduction(root);
	//printf("[li_reduc] ER rest : %d\n", G_td.ER_no_rest);
	
#if 0
	for (int i=0; i<G_td.ER_no; i++) {
		if (G_td.ER_no_array[i]) {
			if (G_td.ERs[i].ER_size < G_td.r.min_size)
				continue;
			save_ER(&G_td.ERs[i], i);
		}
	}
#endif

	/* Tree accumulation */
	int no_union = 0;
	printf("[li_reduc] ER rest : %d\n", G_td.ER_no_rest);
	G_td.ta_algo = tree_accumulation_algo_1;
	ER_un_t *C_union = tree_accumulation(root, &no_union);
	printf("[tr_accum] ER rest : %d\n", no_union);

	/* Print result */
#if 1
	char filepath[100];
	sprintf(filepath,"../../../../../../../LargeFiles/c_impl/0412/[%03d]", G_td.img_id);

	u8 *img_data = (u8 *)malloc(G_td.img->rows*(*G_td.img->step.p)*sizeof(u8));
	ER_un_t *cur = C_union;
	for (int i=0; i<no_union; i++) {
		if (cur->ER->ER_size < G_td.r.min_size)
			continue;
		save_ER(cur->ER, i);
		//plot_ER(&ER_union[i]);
		cur = cur->next;
	}
	free(img_data);
#endif

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
			double percent = 5;
			//IplImage *src = cvLoadImage((path_prefix + path_img).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			//IplImage *src = cvLoadImage("../../../../../Dataset/ICDAR_Robust_Reading/SceneTrialTest/ryoungt_05.08.2002/PICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE);
			IplImage *src = cvLoadImage("PICT0034.JPG", CV_LOAD_IMAGE_GRAYSCALE);
			//IplImage *src = cvLoadImage("SMALL_S.png", CV_LOAD_IMAGE_GRAYSCALE);
			IplImage *dst = cvCreateImage(cvSize((int)((src->width*percent)/100), (int)((src->height*percent)/100) ), src->depth, src->nChannels);
			//IplImage *dst = cvCreateImage(cvSize(200, 200), src->depth, src->nChannels);
			cvResize(src, dst, CV_INTER_LINEAR);
			img = dst;
			//cvNamedWindow("a");
			//cvShowImage("a", dst);
			//cvWaitKey(0);
		}
		printf("Time taken: %.2fs (size %d x %d)\n", (double)(clock() - tStart)/CLOCKS_PER_SEC, img.cols, img.rows); tStart = clock();

		G_td.img = &img;
		G_td.img_id = ii;

		// get ERs
		ER_t *ERs = (ER_t *)malloc(img.rows*img.cols*sizeof(ERs[0]));
		LinkedPoint *pts = (LinkedPoint*)malloc((img.rows*img.cols+1)*sizeof(pts[0]));
		int ER_no = get_ERs(img.data, img.rows, img.cols, *img.step.buf, 0/*2:see debug msg*/, ERs, pts);
		printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC); tStart = clock(); 

		// assign some global variables
		G_td.ERs = ERs;
		G_td.ER_no = ER_no;
		G_td.pts = pts;

		// prepare for getting ER candidates
		G_td.featraw = (featraw_t *)malloc(ER_no*sizeof(featraw_t));
		memset(G_td.featraw, 0, ER_no*sizeof(featraw_t));
		G_td.ER_no_array = (u8 *)malloc(ER_no*sizeof(u8));
		memset(G_td.ER_no_array, 1 , ER_no*sizeof(u8));
		G_td.ER_un = (ER_un_t *)malloc(ER_no*sizeof(ER_un_t));
		memset(G_td.ER_un, 0 , ER_no*sizeof(ER_un_t));

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
#if 0
	int img_cols = 4;
	int img_rows = 4;
	u8* img_data = (u8*)malloc(img_cols*img_rows*sizeof(u8));
	u8* img_ptr = img_data;
	img_ptr[0] =  1; img_ptr[1] =  2; img_ptr[2] =  3; img_ptr[3] =  1;
	img_ptr[4] =  2; img_ptr[5] =  2; img_ptr[6] =  3; img_ptr[7] =  2;
	img_ptr[8] =  3; img_ptr[9] =  3; img_ptr[10] = 3; img_ptr[11] = 2;
	img_ptr[12] = 1; img_ptr[13] = 2; img_ptr[14] = 3; img_ptr[15] = 4;
#elif 1
	int img_cols = 3;
	int img_rows = 3;
	u8* img_data = (u8*)malloc(img_cols*img_rows*sizeof(u8));
	u8* img_ptr = img_data;
	img_ptr[0] = 253; img_ptr[1] = 254; img_ptr[2] = 252;
	img_ptr[3] = 254; img_ptr[4] = 254; img_ptr[5] = 252;
	img_ptr[6] = 252; img_ptr[7] = 252; img_ptr[8] = 253; 
#else
	int img_cols = 5;
	int img_rows = 3;
	u8* img_data = (u8*)malloc(img_cols*img_rows*sizeof(u8));
	u8* img_ptr = img_data;
	img_ptr[0] = 109; img_ptr[1] =  58; img_ptr[2] =  34; img_ptr[3] = 144; img_ptr[4] =  66;
	img_ptr[5] = 205; img_ptr[6] = 205; img_ptr[7] = 205; img_ptr[8] = 132; img_ptr[9] = 181;
	img_ptr[10]= 159; img_ptr[11]= 172; img_ptr[12]= 108; img_ptr[13]= 205; img_ptr[14]= 205;
#endif
	Mat img;
	img.rows = img_rows;
	img.cols = img_cols;
	int img_step = img_cols;

	// get ERs
	ER_t* ERs = (ER_t *)malloc(img_rows*img_cols*sizeof(ERs[0]));
	LinkedPoint* pts = (LinkedPoint*)malloc((img_rows*img_cols+1)*sizeof(pts[0]));
	int ER_no = get_ERs(img_data, img_rows, img_cols, img_step, 0/*2:see debug msg*/, ERs, pts);

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

int main_sample_3(void)
{
	const int ntestsamples = 11;
	CvMat* featureVectorSamples = cvCreateMat(ntestsamples, 2, CV_32F);
    {
        CvMat mat;
        cvGetRows(featureVectorSamples, &mat, 0, 1);
        cvSet(&mat, cvRealScalar(1));
    }
    {
        CvMat mat;
        cvGetRows(featureVectorSamples, &mat, 1, ntestsamples);
        cvSet(&mat, cvRealScalar(0));
    }
    int var_count = featureVectorSamples->cols; // number of single features=variables
    int nsamples_all = featureVectorSamples->rows; // number of samples=feature vectors
	
    CvMat* classLabelResponses = cvCreateMat(nsamples_all, 1, CV_32S);
    {
        CvMat mat;
        cvGetRows(classLabelResponses, &mat, 0, 1);
        cvSet(&mat, cvRealScalar(1));
    }
    {
        CvMat mat;
        cvGetRows(classLabelResponses, &mat, 1, nsamples_all);
        cvSet(&mat, cvRealScalar(-1));
    }

	CvMat* var_type = cvCreateMat(var_count + 1, 1, CV_8U);
    cvSet(var_type, cvScalarAll(CV_VAR_ORDERED)); // Inits all to 0 like the code below
	var_type->data.ptr[var_count] = CV_VAR_CATEGORICAL;

	CvBoost boost;
	boost.train(featureVectorSamples, CV_ROW_SAMPLE, classLabelResponses, 0, 0, var_type, 0, CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0));
	boost.save("./boosttest.xml", "boost");

	CvMat testSamples;
	cvGetRows(featureVectorSamples, &testSamples, 1, 2);
	float ans1 = boost.predict(&testSamples, 0, 0, CV_WHOLE_SEQ, false, true);
	float ans2 = boost.predict(&testSamples);

	return 0;
}


int main(void)
{
	rules_t rules = {10, 0.0019, 0.4562, 0.0100, 0.7989};
	memcpy(&G_td.r, &rules, sizeof(rules_t));

	main_sample_3();
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