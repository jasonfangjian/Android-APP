#include "stdio.h"  
#include "cv.h"  
#include "highgui.h"  
#include "Math.h"
#define max_corners 4

#define C CV_PI /3
int Otsu(IplImage* src);

int main14(int argc, char*argv[])
{
	IplImage* img = cvLoadImage("my.jpg", 0);
	cvNamedWindow("img", 1);
	cvShowImage("img", img);
	IplImage* dst = cvCreateImage(cvGetSize(img), 8, 1);
	int threshold = Otsu(img);//�����䷽����ֵ�ָ�
	printf("threshold = %d\n", threshold);
	cvThreshold(img, dst, threshold, 255, CV_THRESH_BINARY);

	cvNamedWindow("dst", 1);
	cvShowImage("dst", dst);
	CvRect roi = cvRect(30, 30, 120, 120);//ȥ�����ӱ���

	IplImage* img1 = cvCreateImage(cvGetSize(dst), dst->depth, dst->nChannels);
	for (int y = 0; y < img1->height; y++)
	{
		for (int x = 0; x < img1->width; x++)
		{
			CvScalar cs = (255);
			cvSet2D(img1, y, x, cs);
		}
	}
	CvRect roi1 = cvRect(30, 30, 120, 120);
	cvNamedWindow("img1");
	cvShowImage("img1", img1);

	cvSetImageROI(dst, roi);
	cvSetImageROI(img1, roi1);
	cvCopy(dst, img1);
	cvResetImageROI(dst);
	cvResetImageROI(img1);

	cvNamedWindow("result", 1);
	cvShowImage("result", img1);

	IplImage*edge = cvCreateImage(cvGetSize(img1), 8, 1);//canny��Ե���
	int edgeThresh = 1;
	cvCanny(img1, edge, edgeThresh, edgeThresh * 3, 3);
	
	//IplImage* edge = cvLoadImage("bw2.jpg", 0);
	
	cvNamedWindow("canny", 1);
	cvShowImage("canny", edge);
	int count = 0;
	for (int yy = 0; yy < edge->height; yy++)//ͳ�Ʊ�Եͼ���й��ж��ٸ���ɫ���ص�
	{
		for (int xx = 0; xx < edge->width; xx++)
		{
			//CvScalar ss = (255);
			double ds = cvGet2D(edge, yy, xx).val[0];
			if (ds == 0)
				count++;
		}
	}
	int dianshu_threshold = (610*795 - count) / 4;//����ɫ���ص������ķ�֮һ��Ϊhough�任����ֵ
	IplImage* houghtu = cvCreateImage(cvGetSize(edge), IPL_DEPTH_8U, 1);//houghֱ�߱任
	CvMemStorage*storage = cvCreateMemStorage();
	CvSeq*lines = 0;
	int i, j, k, m, n;
	while (true)//ѭ���ҳ����ʵ���ֵ��ʹ��⵽��ֱ�ߵ�������8-12֮��
	{
		lines = cvHoughLines2(edge, storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, dianshu_threshold, 0, 0);
		int line_number = lines->total;
		printf("line_number=%d\n", line_number);
		if (line_number <8)
		{
			dianshu_threshold = dianshu_threshold - 2;
		}
		else if (line_number > 12)
		{
			dianshu_threshold = dianshu_threshold + 1;
		}
		else
		{
			printf("line_number=%d\n", line_number);
			break;
		}
	}

	int A = 10;
	double B = CV_PI / 10;

	while (1)
	{
		for (i = 0; i <lines->total; i++)//�������ǳ������ֱ���޳�
		{
			for (j = 0; j < lines->total; j++)
			{
				if (j != i)
				{
					float*line1 = (float*)cvGetSeqElem(lines, i);
					float*line2 = (float*)cvGetSeqElem(lines, j);
					float rho1 = line1[0];
					float threta1 = line1[1];
					float rho2 = line2[0];
					float threta2 = line2[1];
					if (abs(rho1 - rho2) < A && abs(threta1 - threta2) < B)
						cvSeqRemove(lines, j);
				}
			}
		}
		if (lines->total > 4)//�޳�һȦ�����ֱ�ߵ���������4����ı�A��B������ɾ�����Ƶ�ֱ��
		{
			A = A + 1;
			B = B + CV_PI / 180;
		}
		else
		{
			printf("lines->total=%d\n", lines->total);
			break;
		}
	}




	for (k = 0; k < lines->total; k++)//����ֱ��
	{
		float*line = (float*)cvGetSeqElem(lines, k);
		float rho = line[0];//r=line[0]
		float threta = line[1];//threta=line[1]
		CvPoint pt1, pt2;
		double a = cos(threta), b = sin(threta);
		double x0 = a*rho;
		double y0 = b*rho;
		pt1.x = cvRound(x0 + 100 * (-b));//����ֱ�ߵ��յ����㣬ֱ����ÿһ����Ӧ������ֱ�߷���r=xcos(threta)+ysin(threta);
		pt1.y = cvRound(y0 + 100 * (a));
		pt2.x = cvRound(x0 - 1200 * (-b));
		pt2.y = cvRound(y0 - 1200 * (a));
		cvLine(houghtu, pt1, pt2, CV_RGB(0, 255, 255), 1, 8);
	}
	int num = 0;
	CvPoint arr[8] = { { 0, 0 } };
	for (m = 0; m < lines->total; m++)//����ֱ�ߵĽ���
	{
		for (n = 0; n < lines->total; n++)
		{
			if (n != m)
			{
				float*Line1 = (float*)cvGetSeqElem(lines, m);
				float*Line2 = (float*)cvGetSeqElem(lines, n);
				float Rho1 = Line1[0];
				float Threta1 = Line1[1];
				float Rho2 = Line2[0];
				float Threta2 = Line2[1];
				if (abs(Threta1 - Threta2) > C)
				{
					double a1 = cos(Threta1), b1 = sin(Threta1);
					double a2 = cos(Threta2), b2 = sin(Threta2);
					CvPoint pt;
					pt.x = (Rho2*b1 - Rho1*b2) / (a2*b1 - a1*b2);//ֱ�߽��㹫ʽ
					pt.y = (Rho1 - a1*pt.x) / b1;
					cvCircle(houghtu, pt, 3, CV_RGB(255, 255, 0));
					arr[num++] = pt;//��������걣����һ��������
				}
			}

		}
	}
	printf("num=%d\n", num);
	printf("arr[0].x=%d\n", arr[0].x);
	printf("arr[0].y=%d\n", arr[0].y);
	printf("arr[1].x=%d\n", arr[1].x);
	printf("arr[1].y=%d\n", arr[1].y);
	printf("arr[2].x=%d\n", arr[2].x);
	printf("arr[2].y=%d\n", arr[2].y);
	printf("arr[3].x=%d\n", arr[3].x);
	printf("arr[3].y=%d\n", arr[3].y);
	printf("arr[4].x=%d\n", arr[4].x);
	printf("arr[4].y=%d\n", arr[4].y);
	printf("arr[5].x=%d\n", arr[5].x);
	printf("arr[5].y=%d\n", arr[5].y);
	printf("arr[6].x=%d\n", arr[6].x);
	printf("arr[6].y=%d\n", arr[6].y);
	printf("arr[7].x=%d\n", arr[7].x);
	printf("arr[7].y=%d\n", arr[7].y);

	CvPoint arr1[8] = { { 0, 0 } };//���ظ��Ľǵ��޳�
	int num1 = 0;
	for (int r = 0; r < 8; r++)
	{
		int s = 0;
		for (; s < num1; s++)
		{
			if (abs(arr[r].x - arr1[s].x) <= 2 && abs(arr[r].y - arr1[s].y) <= 2)
				break;
		}
		if (s == num1)
		{
			arr1[num1] = arr[r];
			num1++;
		}

	}


	printf("num1=%d\n", num1);
	printf("arr1[0].x=%d\n", arr1[0].x);
	printf("arr1[0].y=%d\n", arr1[0].y);
	printf("arr1[1].x=%d\n", arr1[1].x);
	printf("arr1[1].y=%d\n", arr1[1].y);
	printf("arr1[2].x=%d\n", arr1[2].x);
	printf("arr1[2].y=%d\n", arr1[2].y);
	printf("arr1[3].x=%d\n", arr1[3].x);
	printf("arr1[3].y=%d\n", arr1[3].y);
	printf("arr1[4].x=%d\n", arr1[4].x);
	printf("arr1[4].y=%d\n", arr1[4].y);
	printf("arr1[5].x=%d\n", arr1[5].x);
	printf("arr1[5].y=%d\n", arr1[5].y);
	printf("arr1[6].x=%d\n", arr1[6].x);
	printf("arr1[6].y=%d\n", arr1[6].y);
	printf("arr1[7].x=%d\n", arr1[7].x);
	printf("arr1[7].y=%d\n", arr1[7].y);

	for (int w = 0; w < 4; w++)
	{
		CvPoint ps;
		ps = arr1[w];
		cvCircle(img, ps, 3, CV_RGB(255, 0, 0));
	}
	cvNamedWindow("img", 1);
	cvShowImage("img", img);
	cvNamedWindow("houghtu", 1);
	cvShowImage("houghtu", houghtu);
	cvWaitKey(-1);

	cvReleaseImage(&img);
//	cvReleaseImage(&dst);


	cvDestroyWindow("dst");
	return 0;
}


int Otsu(IplImage* src)
{
	int height = src->height;
	int width = src->width;

	//histogram    
	float histogram[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		unsigned char* p = (unsigned char*)src->imageData + src->widthStep * i;
		for (int j = 0; j < width; j++)
		{
			histogram[*p++]++;
		}
	}
	//normalize histogram    
	int size = height * width;
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / size;
	}

	//average pixel value    
	float avgValue = 0;
	for (int i = 0; i < 256; i++)
	{
		avgValue += i * histogram[i];  //����ͼ���ƽ���Ҷ�  
	}

	int threshold;
	float maxVariance = 0;
	float w = 0, u = 0;
	for (int i = 0; i < 256; i++)
	{
		w += histogram[i];  //���赱ǰ�Ҷ�iΪ��ֵ, 0~i �Ҷȵ�����(��������ֵ�ڴ˷�Χ�����ؽ���ǰ������) ��ռ����ͼ��ı���  
		u += i * histogram[i];  // �Ҷ�i ֮ǰ������(0~i)��ƽ���Ҷ�ֵ�� ǰ�����ص�ƽ���Ҷ�ֵ  

		float t = avgValue * w - u;
		float variance = t * t / (w * (1 - w));
		if (variance > maxVariance)
		{
			maxVariance = variance;
			threshold = i;
		}
	}

	return threshold;
}
