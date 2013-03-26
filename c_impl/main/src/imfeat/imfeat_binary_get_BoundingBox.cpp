
#include "../../include/system.h"
#include "../../include/imfeat.h"
#include "imfeat_internal.h"
#include <stdio.h>
#include <stdlib.h>

void get_BoundingBox(IN p8_t *pt, IN p4_t *feat_in, OUT p4_t *feat_out)
{
	LinkedPoint *pts = (LinkedPoint *)pt->val[0];
	int no_fr = pt->val[3];
	int pt_all_start_idx = pt->val[6];
	int pt_all_no = pt->val[7];

	int l,t,r,b;
	LinkedPoint *cur = &pts[pt_all_start_idx];
	if (no_fr == 0) {
		l = r = cur->pt.x;
		t = b = cur->pt.y;
	} else {
		// accumulate multiple input feature
		l = feat_in[0].val[0];
		t = feat_in[0].val[1];
		r = feat_in[0].val[2];
		b = feat_in[0].val[3];
		for (int i=1; i<no_fr; i++) {
			l = MIN(l, feat_in[i].val[0]);
			t = MIN(t, feat_in[i].val[1]);
			r = MAX(r, feat_in[i].val[2]);
			b = MAX(b, feat_in[i].val[3]);
		}
	}

	// preprocess: label pixel
	imfeat_util_label_pixels(1, pt);

	// calc bounding box
	for (int i=0; i<pt_all_no; i++, cur=cur->next) {
		if (PXL_IS_NEW(cur)) {
			l = MIN(l, cur->pt.x);
			t = MIN(t, cur->pt.y);
			r = MAX(r, cur->pt.x);
			b = MAX(b, cur->pt.y);
		}
	}

	// postprocess: clear pixel
	imfeat_util_label_pixels(0, pt);

	// output feature
	feat_out->val[0] = l;
	feat_out->val[1] = t;
	feat_out->val[2] = r;
	feat_out->val[3] = b;
}
