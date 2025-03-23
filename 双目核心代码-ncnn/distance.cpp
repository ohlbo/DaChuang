#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <math.h> 
#include "camyam.h"
#include <opencv2/core/utils/logger.hpp>



using namespace std;
using namespace cv;

struct CallbackParams {
	bool selectObject;
	Rect selection;
	Mat xyz;
	Mat rectifyImageL;
	Mat rectifyImageR;
	Mat Q;
	int blockSize=6;
	int n=5;
};

void calclt_dist(int x, int y, void* data) {
    CallbackParams* params = static_cast<CallbackParams*>(data);
    
    // 检查坐标是否越界
    if (x < 0 || x >= params->xyz.cols || y < 0 || y >= params->xyz.rows) {
        cerr << "Error: Coordinates out of bounds." << endl;
        return;
    }

    // 获取点的深度值
    Vec3f point3 = params->xyz.at<Vec3f>(Point(x, y));
    
    // 检查深度值是否有效
    if (std::isnan(point3[0]) || std::isnan(point3[1]) || std::isnan(point3[2])) {
        cerr << "Error: Invalid depth values at (" << x << ", " << y << ")." << endl;
        return;
    }
    
    // 计算距离
    float d = sqrt(point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2]);
    d = d / 1000.0;  //转换为米
	int precision = 2;	//动态调节输出格式
    std::cout << "Distance: " << std::fixed << std::setprecision(precision) << d << " m" << std::endl;
}

static void clickMouse(int event, int x, int y, int, void* data)
{
	CallbackParams* params = static_cast<CallbackParams*>(data);
	static Point origin;
	if (params->selectObject)
	{
		params->selection.x = MIN(x, origin.x);
		params->selection.y = MIN(y, origin.y);
		params->selection.width = std::abs(x - origin.x);
		params->selection.height = std::abs(y - origin.y);
	}
	float d;
	Vec3f point3;
	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		params->selection = Rect(x, y, 0, 0);
		params->selectObject = true;
		//cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		//Mat xyz;
		point3 = params->xyz.at<Vec3f>(origin);
		//point3[0];
		//cout << "像素坐标： " << origin << endl;
		//cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
		//cout << "世界坐标：" << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
		d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
		d = sqrt(d);   //mm
	   // cout << "距离是:" << d << "mm" << endl;

		//d = d / 10.0;   //cm
		//cout << "距离:" << d << "cm" << endl;

		d = d / 1000.0;   //m
		cout << "距离是:" << d << "m" << endl;

		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		params->selectObject = false;
		if (params->selection.width > 0 && params->selection.height > 0)
			break;
	}
}

void get_depth(int ,int x,int y, void* data) {
	CallbackParams* params = static_cast<CallbackParams*>(data);
	//创建sgbm对象
	int n = params->n;
	int blockSize = params->blockSize;
	Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16 * n, blockSize);
	//SGBM参数设置
	{
		int P1 = 8 * 3 * blockSize * blockSize;
		int P2 = 32 * 3 * blockSize * blockSize;
		sgbm->setP1(P1);
		sgbm->setP2(P2);
		sgbm->setPreFilterCap(1);
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleRange(100);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setDisp12MaxDiff(-1);
		//sgbm->setNumDisparities(1);
		sgbm->setMode(cv::StereoSGBM::MODE_HH);
	}

	//计算视差
	Mat disp, disp8, disColor, disColorMap;
	sgbm->compute(params->rectifyImageL, params->rectifyImageR, disp);//输入灰度图，输出视差

	disColor = Mat(disp.rows, disp.cols, CV_8UC1);
	normalize(disp, disColor, 0, 255, NORM_MINMAX, CV_8U);
	applyColorMap(disColor, disColorMap, COLORMAP_JET);
	reprojectImageTo3D(disp, params->xyz, params->Q, true);
	params->xyz *= 16;
	//显示深度图
	cv::circle(disColorMap, cv::Point(x, y), 5, Scalar(255, 255, 255), -1);
	imshow("depth", disColorMap);


}

void printword() {
    std::cout << "Hello, World!" << std::endl;
}

/*
int main() {

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	
	//单个摄像头的分辨率  
	const int imageWidth = 640;
	const int imageHeight = 480;

	//单个相机的分辨率
	Size imageSize = Size(imageWidth, imageHeight);


	Mat rgbImageL, grayImageL;//彩色、灰色左视图
	Mat rgbImageR, grayImageR;
	Mat rectifyImageL, rectifyImageR;//校正后左右视图

	//定义切割区域
	Rect m_l_select;
	Rect m_r_select;

	//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
	Rect validROIL;
	Rect validROIR;

	//映射表  
	Mat mapLx, mapLy, mapRx, mapRy;
	//校正旋转矩阵R，投影矩阵P 重投影矩阵Q
	Mat Rl, Rr, Pl, Pr, Q;
	Mat R;//R 旋转矩阵

	//立体校正参数计算
	Rodrigues(rec, R);
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);//-1为默认自由缩放参数；参数解释：https://baike.baidu.com/item/stereoRectify/1594417?fr=ge_ala
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);

	//获取左右图像
	m_l_select = Rect(0, 0, imageWidth, imageHeight);
	m_r_select = Rect(imageWidth, 0, imageWidth, imageHeight);
	VideoCapture cap(0);//打开摄像头（输入摄像头的索引）
	cap.set(CAP_PROP_FRAME_WIDTH, imageWidth*2);
	cap.set(CAP_PROP_FRAME_HEIGHT, imageHeight);


	if (!cap.isOpened()) {
		cout << "Error: Could not open the camera." << endl;
		return -1;
	}
	CallbackParams params;
	params.blockSize=6;
	params.n=5;
	params.selectObject = false;
	params.Q = Q;
	namedWindow("depth", WINDOW_AUTOSIZE);
	
	while (true)
	{
		Mat frame;
		cap.read(frame);
		if (frame.empty()) break;

		 // 分割左右图像
		Mat frameL = frame(m_l_select);
		Mat frameR = frame(m_r_select);

		if (frameL.empty() || frameR.empty()) {
			cout << "Error: Could not grab a frame." << endl;
			break;
		}

		// 转换为灰度图
		cvtColor(frameL, grayImageL, COLOR_BGR2GRAY);
		cvtColor(frameR, grayImageR, COLOR_BGR2GRAY);

		// 进行矫正
		remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

		// 设置参数
		params.rectifyImageL = rectifyImageL;
		params.rectifyImageR = rectifyImageR;

		// 显示深度图
		get_depth(0,240,240, &params);
		
		//计算坐标的距离
		calclt_dist(240,240,&params);


		// 按下ESC退出
		if (waitKey(30) == 27) break;

	}
}
#endif
*/
