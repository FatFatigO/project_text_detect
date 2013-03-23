
#ifndef IMFEAT_INTERNAL_H_
#define IMFEAT_INTERNAL_H_

/*    Ex: Merge upper ER1(1,2,2,2) and lower ER2(1,2) to ER3
 *
 * (gray level)   (label for ER1) | (label for ER2) = (union labels)  
 *    1 2 3 7        1 1 D X          D D D X        (1O)(1O)( O)( X)       "PXL_DIF|PXL_ORI" "PXL_DIF|PXL_ORI" "PXL_DIF" "    "
 *    2 2 3 6        1 1 D X          D D D X        (1O)(1O)( O)( X)       "PXL_DIF|PXL_ORI" "PXL_DIF|PXL_ORI" "PXL_DIF" "    "
 *    3 3 3 5        D D D X          D D D X        ( O)( O)( O)( X)       "PXL_DIF"         "PXL_DIF"         "PXL_DIF" "    "
 *    1 2 3 4        D D D X          2 2 D X        (2O)(2O)( O)( X)       "PXL_DIF|PXL_ORI" "PXL_DIF|PXL_ORI" "PXL_DIF" "    "
 *
 *    where: (in union labels)
 *      1&2 will be labeled PXL_ORI
 *        D will be labeled PXL_DIF
 *        X will be labeled "nothing"
 *      pixels only labeled with PXL_DIF will be regarded as PXL_NEW
 */
#define PXL_ORI			0x001
#define PXL_DIF			0x002
#define PXL_ACU			0x004
#define PXL_IMG_EDG		0x010
#define PXL_ERS_EDG		0x020
#define ROW_DIF			0x100
#define ROW_NEW			0x200

#define PXL_IS_ORI(x)		((x->flag&PXL_ORI)>0)
#define PXL_IS_NEW(x)		(((x->flag&PXL_DIF)>0)&&!((x->flag&PXL_ORI)>0))
#define PXL_IS_ACU(x)		((x->flag&PXL_ACU)>0)
#define PXL_IS_IMG_EDG(x)	((x->flag&PXL_IMG_EDG)>0)
#define PXL_IS_ERS_EDG(x)	((x->flag&PXL_ERS_EDG)>0)
#define ROW_HAS_DIF(x)		((x->prev->flag&ROW_DIF)>0)

#define PXL_GO_ORI(x)		(x->flag|=PXL_ORI)
#define PXL_GO_DIF(x)		(x->flag|=PXL_DIF)
#define PXL_GO_ACU(x)		(x->flag|=PXL_ACU)
#define ROW_SEE_DIF(x)		((x->prev->flag|=ROW_DIF)>0)

extern void imfeat_util_label_pixels(int label, p8_t *pt);

#endif