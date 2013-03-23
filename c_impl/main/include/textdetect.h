
#ifndef TEXTDETECT_H_
#define TEXTDETECT_H_

#include "system.h"
#include <opencv2/core/core.hpp>
using namespace cv;

typedef struct featraw_t
{
	/* public used */
	p4_t BB;		// bounding box (l,t,r,b)
	p1_t PR;		// perimeter
	p1_t EN;		// euler no
	p1_t HC;		// horizontal crossing
	int *HC_buf;	// buf addr backup for memory relase
} featraw_t;

// Global variable structure
typedef struct G_textdetect_t
{
	Mat *img;            // current image
	ER_t *ERs;           // ERs
	int ER_no;           // ER no
	LinkedPoint *pts;    // points
	featraw_t *featraw;  // raw feature for each ER
} G_textdetect_t;

extern G_textdetect_t G_td;

#endif
