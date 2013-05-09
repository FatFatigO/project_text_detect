
#ifndef TEXTDETECT_H_
#define TEXTDETECT_H_

#include "system.h"
#include <opencv2/core/core.hpp>
using namespace cv;

#define INVALID_ER(v)	(v->size=0xDEADBEEF)
#define ISVALID_ER(v)	(v->size==0xDEADBEEF)

typedef struct featraw_t
{
	/* public used */
	p4_t BB;		// bounding box (l,t,r,b)
	p1_t PR;		// perimeter
	p1_t EN;		// euler no
	p1_t HC;		// horizontal crossing
	int *HC_buf;	// buf addr backup for memory relase
} featraw_t;

typedef struct ER_un_t
{
	ER_t *ER;
	ER_un_t *prev;
	ER_un_t *next;
} ER_un_t;

typedef struct rules_t
{
	int min_size;// = 30;
	double min_w_reg2img_ratio;// = 0.0019;
	double max_w_reg2img_ratio;// = 0.4562;
	double min_h_reg2img_ratio;// = 0.0100;
	double max_h_reg2img_ratio;// = 0.7989;
} rules_t;

// Global variable structure
typedef struct G_textdetect_t
{
	Mat *img;            // current image
	int img_id;          // current image id
	ER_t *ERs;           // ERs
	int ER_no;           // ER no
	int ER_no_rest;      // ER no rest
	u8 *ER_no_array;     // ER no array
	ER_un_t *ER_un;      // ER union after tree accumulation
	LinkedPoint *pts;    // points
	featraw_t *featraw;  // raw feature for each ER
	rules_t r;           // rule constants
	void *boost;         // boost classifier

	bool (*lr_algo)(ER_t *, ER_t *);              // linear-reduction algo
	bool (*ta_algo)(ER_t *, int, ER_un_t *);    // tree-accumulation algo

} G_textdetect_t;

extern G_textdetect_t G_td;

#endif
