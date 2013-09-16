
#ifndef UTIL_H_
#define UTIL_H_

extern CvRect rect_intersect(CvRect r1, CvRect r2);
extern void rect_accumulate_start(CvRect r);
extern void rect_accumulate_end(void);
extern void rect_accumulate_rect(CvRect r);
extern double rect_accumulate_get_percent(void);

#endif