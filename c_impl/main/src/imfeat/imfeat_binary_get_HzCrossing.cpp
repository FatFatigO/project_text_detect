
#include "../../include/system.h"
#include "../../include/imfeat.h"
#include "imfeat_internal.h"
#include <stdio.h>
#include <stdlib.h>

void get_HzCrossing(IN p8_t *pt, IN p1_t *feat_in, OUT p1_t *feat_out)
{
	LinkedPoint *pts = (LinkedPoint *)pt->val[0];
	int img_rows = pt->val[1];
	int img_cols = pt->val[2];
	int no_fr = pt->val[3];
	int pt_all_start_idx = pt->val[6];
	int pt_all_no = pt->val[7];
	int *feat_o = (int *)feat_out->val[0];

	// proprocess: label pixel
	imfeat_util_label_pixels(1, pt);
	
	// go back to (0,0) first
	LinkedPoint *row_1st_pt = pts;
	int shift_l = row_1st_pt->pt.x;
	int shift_t = row_1st_pt->pt.y;
	for (int i=0; i<shift_l; i++) row_1st_pt = row_1st_pt->l;
	for (int i=0; i<shift_t; i++) row_1st_pt = row_1st_pt->t;

	// calc hohrizontal crossing
	for (int h=0; h<img_rows; h++, row_1st_pt=row_1st_pt->b) {
		LinkedPoint *cur = row_1st_pt;
		int q = 0;
		if (ROW_HAS_DIF(cur)) {
			while (!PXL_IS_IMG_EDG(cur)) {
				if (PXL_IS_NEW(cur)) {
					if (!PXL_IS_ACU(cur->l) && !PXL_IS_ACU(cur->r))
						q = q + 2;
					if (PXL_IS_ACU(cur->l) && PXL_IS_ACU(cur->r))
						q = q - 2;
					PXL_GO_ACU(cur);
				}
				cur = cur->r;
			}
		}
		// accumulate multiple input feature
		for (int i=0; i<no_fr; i++) {
			int *a = (int *)feat_in[i].val[0];
			*feat_o += a[h];
		}
		*feat_o += q;
		feat_o++;
	}

	// postprocess: clear pixel
	imfeat_util_label_pixels(0, pt);
}
