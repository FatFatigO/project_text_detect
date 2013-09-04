
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
	/* required parameter from user */
	int text_is_darker;         // = 1 or 0
	int tree_accum_algo;        // = 1 or 2 or 3
	double min_reg2img_ratio;   // = 0.001;
	double max_reg2img_ratio;   // = 0.5
	double min_ar;              // = 1.2
	double max_ar;              // = 0.7
	double small_ar_pnty_coef;  // = 0.08
	double large_ar_pnty_coef;  // = 0.03
	/* not used so far */
	double min_w_reg2img_ratio;// = 0.0019;
	double max_w_reg2img_ratio;// = 0.4562;
	double min_h_reg2img_ratio;// = 0.0100;
	double max_h_reg2img_ratio;// = 0.7989;
	double min_postp_delta;    // = 0.1;
	double min_postp_value;    // = 0.2;
	/* for internal use */
	int min_size;
	int max_size;
} rules_t;

// Global variable structure
typedef struct G_textdetect_t
{
	Mat *img;            // current image
	int img_id;          // current image id
	char channel;        // current channeld
	char *output_path;   // output path
	ER_t *ERs;           // ERs
	int ER_no;           // ER no
	int ER_no_rest;      // ER no rest
	u8 *ER_no_array;     // ER no array
	ER_un_t *ER_un;      // ER union after tree accumulation
	LinkedPoint *pts;    // points
	featraw_t *featraw;  // raw feature for each ER
	rules_t r;           // rule constants
	void *boost;         // boost classifier
	u8 *hc1;             // used for horizontal crossing feature
	u8 *hc2;
	u8 *hc3;

	bool (*lr_algo)(ER_t *, ER_t *);              // linear-reduction algo
	bool (*ta_algo)(ER_t *, int, ER_un_t *);    // tree-accumulation algo

} G_textdetect_t;

extern G_textdetect_t G_td;

extern void text_detect(Mat *img, int text_is_darker, char *output_path, int algo);

#endif
