
#include "../../include/system.h"
#include "../../include/imfeat.h"
#include "imfeat_internal.h"
#include <stdio.h>
#include <stdlib.h>

void get_Perimeter(IN p8_t *pt, IN p1_t *feat_in, OUT p1_t *feat_out)
{
	LinkedPoint *pts = (LinkedPoint *)pt->val[0];
	int no_fr = pt->val[3];
	int pt_all_start_idx = pt->val[6];
	int pt_all_no = pt->val[7];

	// accumulate multiple input feature
	int p = 0;
	for (int i=0; i<no_fr; i++) {
		p += feat_in[i].val[0];
	}

	// proprocess: label pixel
	imfeat_util_label_pixels(1, pt);
	
	// calc perimeter
	LinkedPoint *cur = &pts[pt_all_start_idx];
	for (int i=0; i<pt_all_no; i++, cur=cur->next) {
		if (PXL_IS_NEW(cur)) {
			// process each new added pixels here
			// (1) calc num of adjacent edge q with accumulated map
			int q = 0;
			if (cur->l)
				if PXL_IS_ACU(cur->l)
					q = q + 1;
			if (cur->t)
				if PXL_IS_ACU(cur->t)
					q = q + 1;
			if (cur->r)
				if PXL_IS_ACU(cur->r)
					q = q + 1;
			if (cur->b)
				if PXL_IS_ACU(cur->b)
					q = q + 1;
			// (2) calc edge no change: 4 - 2{q:qAp^C(q)<=C(p)}
			p = p + (4 - 2*q);
			// (3) set cur pt as 1
			PXL_GO_ACU(cur);
		}
	}

	// postprocess: clear pixel
	imfeat_util_label_pixels(0, pt);

	// output feature
	feat_out->val[0] = p;
}
