#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
using namespace std;
using namespace cv;


const string srcImg1 = "test/test8_fuse.png";   //黑白的
const string srcImg2 = "test/test8.jpg";          //原图
Mat image1 = imread(srcImg1, 0);
Mat image2 = imread(srcImg2, 0);





class LineFinder
{
private:
	cv::Mat img;
	//包含被检测直线的端点的向量
	std::vector<cv::Vec4i> lines;

	//累加器分辨率参数
	double deltaRho;
	double deltaTheta;

	//确认直线之前必须受到的最小投票数
	int minVote;

	//直线的最小长度
	double minLength;
	//直线上允许的最大空隙
	double maxGap;
public:
	LineFinder() :deltaRho(1), deltaTheta(3.14159 / 180), minVote(10), minLength(0.0), maxGap(0.0) {}
	void setAccResolution(double dRho, double dTheta)
	{
		deltaRho = dRho;
		deltaTheta = dTheta;
	}
	void setminVote(int minv)
	{
		minVote = minv;
	}
	void setLengthAndGap(double length, double gap)
	{
		minLength = length;
		maxGap = gap;
	}
	std::vector<cv::Vec4i> findLines(cv::Mat& binary)
	{
		lines.clear();
		cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
		return lines;
	}
	void drawDetectedLines(cv::Mat& image, cv::Scalar color = cv::Scalar(0, 255, 255))
	{
		std::vector<cv::Vec4i>::const_iterator it = lines.begin();
		while (it != lines.end())
		{
			cv::Point pt1((*it)[0], (*it)[1]);
			cv::Point pt2((*it)[2], (*it)[3]);
			cv::line(image, pt1, pt2, Scalar(0, 0, 255),5);
			it++;
		}
	}
};

class Perspective{
private:
	const Mat srcImg;   //原图
	Mat dstImg;         //校正图
	Mat grayImg;        //灰度图
	int top;            //处理点上边缘
	int bottom;         //处理点下边缘
	int img_height;
	int img_width;
public:
	Perspective(const Mat &image, Mat &result);
	const Mat Calibration();   //校正过程函数
	cv::Vec3f EdgeExtracting(vector<cv::Vec4i> lines, bool direction);    //边缘提取；direction：0=左侧，1=右侧；返回四元组：直线上两点坐标
};

Perspective::Perspective(const Mat &image, Mat &result) :srcImg(image) {
	if (!srcImg.data)       //检测是否读图成功
		cout << "Failed to read data!" << endl;
	dstImg = result;
	dstImg.create(image.rows, image.cols, image.type());
	Mat tempImg;
	//cvtColor(srcImg, tempImg, CV_RGB2GRAY);   //灰度化
	//equalizeHist(tempImg, grayImg);            //直方图均衡化
	srcImg.copyTo(grayImg);
	img_height = srcImg.rows;
	img_width = srcImg.cols;
	top = 0;   
	bottom = img_height - 1;
}

cv::Vec3f Perspective::EdgeExtracting(vector<cv::Vec4i> lines,  bool direction) {
	int n = lines.size();    //直线总数
	int border=srcImg.cols,current=0;    //border最小边界距离，current当前边
	int temp;
	double k,current_k=0;  //斜率
	for (int i = 0; i < n; i++) {
		if (lines[i][0] == lines[i][2]|| lines[i][1] == lines[i][3])   //去除垂直线
			break;
		if (lines[i][1]>lines[i][3])     //确保方向相同
			k = double(lines[i][1] - lines[i][3]) / (lines[i][0] - lines[i][2]);
		else
			k = double(lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0]);
		if ((direction && k < tan(120)) || (!direction && k > tan(60))) {
			if(direction)
				temp = srcImg.cols - (lines[i][0] > lines[i][2]) ? lines[i][0] : lines[i][2];   //取x坐标大的点,计算边界距离
			else
				temp = (lines[i][0] < lines[i][2]) ? lines[i][0] : lines[i][2];   //取x坐标小的点,计算边界距离
			if (border > temp) {
				border = temp;
				current = i;     //记录当前最小边界距离的边
				current_k = k;   //记录当前边斜率
			}
		}		
	}
	cv::Vec3f edge;
	edge[0] = current_k;
	edge[1] = lines[current][0];
	edge[2] = lines[current][1];
	return edge;
}

const Mat Perspective::Calibration() {
	LineFinder cFinder;
	cFinder.setLengthAndGap(80, 20);
	cFinder.setminVote(60);
	Mat contours;   //边缘图
	cv::Canny(grayImg, contours, 350, 600);   //边缘检测，阈值
	cv::imshow("Canny Image", contours);
	cv::Vec3f l_line = EdgeExtracting(cFinder.findLines(contours),  0);   //得到左侧直线
	cv::Vec3f r_line = EdgeExtracting(cFinder.findLines(contours),  1);   //得到右侧直线
	//cv::Mat findLines;
	//findLines= imread("1.jpg");
	//cFinder.drawDetectedLines(findLines);
	//cv::imshow("detected result", findLines);
	Mat lineImage(contours.size(), CV_8U, cv::Scalar(0));
	cv::line(lineImage, cv::Point(l_line[1], l_line[2]), cv::Point(l_line[1] + 100, l_line[2] + 100 * l_line[0]), cv::Scalar(255), 3);
	cv::line(lineImage, cv::Point(r_line[1], r_line[2]), cv::Point(r_line[1] + 100, r_line[2] + 100 * r_line[0]), cv::Scalar(255), 3);
	cv::imshow("detected line", lineImage);
	double l_t = 1 / l_line[0];   //斜率倒数
	double r_t = 1 / r_line[0];
	int l_x = l_line[1];    //左点
	int l_y = l_line[2];
	int r_x = r_line[1];    //右点
	int r_y = r_line[2];
	vector<Point2f> corners(4);
	corners[0] = Point2f((top - l_y)*l_t + l_x, top);           //直线延伸到上边缘的点
	corners[1] = Point2f((top - r_y)*r_t + r_x, top);
	corners[2] = Point2f((bottom - l_y)*l_t + l_x, img_height - 1);   //直线延伸到下边缘的点
	corners[3] = Point2f((bottom - r_y)*r_t + r_x, img_height - 1);
	vector<Point2f> corners_trans(4);
	corners_trans[0] = Point2f((top - l_y)*l_t + l_x, top);     //下边缘向上边缘校正
	corners_trans[1] = Point2f((top - r_y)*r_t + r_x, top);
	corners_trans[2] = Point2f((top - l_y)*l_t + l_x, img_height - 1);
	corners_trans[3] = Point2f((top - r_y)*r_t + r_x, img_height - 1);
	Mat transform = getPerspectiveTransform(corners, corners_trans);    //获取转换矩阵
	warpPerspective(srcImg, dstImg, transform, srcImg.size(), INTER_LINEAR);    //透视校正
	return dstImg;
}


int main12()
{
	Mat dst = Mat::zeros(image2.rows, image2.cols, CV_8UC1);
	resize(image1, dst, dst.size());
	imshow("尺寸调整之前", image1);
	imshow("尺寸调整之后", dst);            //图像缩放
	vector<vector<Point>> contours;
	Mat Biny;
	threshold(dst, Biny, 30, 200, CV_THRESH_BINARY);
	Mat conVt(Biny.size(), CV_8UC1);
	//cvtColor(Biny, conVt, CV_BGR2GRAY);
	Biny.convertTo(conVt, CV_8UC1);
	findContours(conVt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat result(conVt.size(), CV_8U, Scalar(255));
	drawContours(result, contours, -1, Scalar(0), 2);
	imshow("边缘增强", result);
	imwrite("my.jpg", result);
	Mat image,result2;
	image2.copyTo(image);
	Mat contours2;   //边缘图
	result.copyTo(contours2);
	/*LineFinder cFinder;
	cFinder.setLengthAndGap(80, 20);
	cFinder.setminVote(60);
	cv::Canny(image, contours2, 350, 400);
	std::vector<cv::Vec4i> lines = cFinder.findLines(contours2);
	cFinder.drawDetectedLines(image);
	cv::imshow("detected result", image);
	*/

	std::vector<cv::Vec2f> lines;
	cv::HoughLines(contours2, lines,
		1, 3.14 / 180,//步长
		60);//最小投票数
	std::vector<cv::Vec2f>::const_iterator it = lines.begin();
	while (it != lines.end())
	{
		float rho = (*it)[0];
		float theta = (*it)[1];
		if (theta<3.14 / 4.0 || theta>3.0*3.14 / 4.0)
		{
			//直线与第一行的交叉点
			cv::Point pt1(static_cast<int>(rho / cos(theta)), 0.0);
			//直线与最后一行的交叉点
			cv::Point pt2(rho / cos(theta) - image.rows*sin(theta) / cos(theta), image.rows);
			cv::line(image, pt1, pt2, cv::Scalar(255, 255, 255), 1);
		}
		else
		{
			cv::Point pt1(0, rho / sin(theta));
			cv::Point pt2(image.cols, rho / sin(theta) - image.cols*cos(theta) / sin(theta));
			cv::line(image, pt1, pt2, cv::Scalar(255, 255, 255), 1);
		}
		it++;
	}
	cv::imshow("result", image);

	waitKey();
	return 0;
}

/*
void main()
{
	Mat image, result;
	image = imread("1.jpg");
	if (!image.data) {
		cout << "Error!" << endl;
		return;
	}
	Perspective dist(image, result);
	result = dist.Calibration();
	namedWindow("Original Image");
	imshow("Original Image", image);
	namedWindow("Output Image");
	imshow("Output Image", result);
	waitKey(0);
}
*/