
#include "../../include/system.h"
#include "../../include/imfeat.h"
#include "imfeat_internal.h"

// label = 1: label pixel flag
// label = 0: clear pixel flag
static void label_pixels(int label, p5_t *param)
{
	LinkedPoint *pts = (LinkedPoint *)param->val[0];
	int pt_ori_start = param->val[1];
	int pt_ori_size = param->val[2];
	int pt_all_start = param->val[3];
	int pt_all_size = param->val[4];

	// prerocess: label pixels
	LinkedPoint *cur = &pts[pt_all_start];
	int ori = 0, ori_cnt = 0, saw_new = 0;
	for (int i=0; i<pt_all_size; i++, cur=cur->next) {
		if (cur->pt_order==pt_ori_start) ori = 1;

		if (ori==1) {
			if (label==1) {
				PXL_GO_ORI(cur);
				PXL_GO_ACU(cur);
			} else {
				cur->flag = 0;
				cur->prev->flag = 0;
			}
			ori_cnt ++;
		}							   
		if (ori==0) {
			if (label==1) {
				PXL_GO_DIF(cur);
				ROW_SEE_DIF(cur);
			} else {
				cur->flag = 0;
				cur->prev->flag = 0;
			}
		}

		if (ori_cnt==pt_ori_size) ori = 0;
	}
}

void imfeat_util_label_pixels(int label, p8_t *pt)
{
	int no_fr = pt->val[3];
	if (no_fr==0) {
		p5_t param;
		param.val[0] = pt->val[0];
		param.val[1] = -1; // let org be out of range so that "ori=0"
		param.val[2] = -1; // let org be out of range so that "ori=0"
		param.val[3] = pt->val[6];
		param.val[4] = pt->val[7];
		label_pixels(label, &param);
	} else {
		for (int i=0; i<no_fr; i++) {
			p5_t param;
			param.val[0] = pt->val[0];
			param.val[1] = *((u32 *)(pt->val[4]) + i);
			param.val[2] = *((u32 *)(pt->val[5]) + i);
			param.val[3] = pt->val[6];
			param.val[4] = pt->val[7];
			label_pixels(label, &param);
		}
	}
}
