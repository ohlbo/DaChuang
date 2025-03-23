#include <stdlib.h>
#include <unistd.h>

#include "layer.h"
#include "net.h"
 
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <sys/time.h>
 
#include <iostream>  
#include <chrono>  
#include <opencv2/opencv.hpp>  

#include <wiringPi.h>

#include <wiringSerial.h>

#include <math.h> 
#include "camyam.h"
#include <opencv2/core/utils/logger.hpp>

#include <thread>
#include <atomic>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <netinet/in.h>

#include <cstdlib>  // 包含 system() 函数的头文件

#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstring>

#include <arpa/inet.h>
#include <nlohmann/json.hpp>

#include <cmath> 

using namespace std;  
using namespace cv;  
using namespace std::chrono;  

std::mutex data_mutex;
double distances = 0.0;  // 存储实时距离数据


#define SERIAL_DEVICE "/dev/ttyAMA0"  // Raspberry Pi 
#define BAUD_RATE 9600  // 

#define SHM_KEY 1234
#define WIDTH 640
#define HEIGHT 480
#define CHANNELS 3
#define IMG_SIZE (WIDTH * HEIGHT * CHANNELS)  // 图像数据大小
#define META_SIZE 256  // 额外的检测结果存储空间
#define SHM_SIZE (IMG_SIZE + META_SIZE)  // 总共享内存大小
#define FOV_VERTICAL 67
#define CAMERA_ANGLE 60

// 0 : FP16
// 1 : INT8
#define USE_INT8 1
 
// 0 : Image
// 1 : Camera
#define USE_CAMERA 1
 
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
 
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
 
static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
 
    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;
 
        while (faceobjects[j].prob < p)
            j--;
 
        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);
 
            i++;
            j--;
        }
    }
 
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
 
static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;
 
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}
 
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
 
    const int n = faceobjects.size();
 
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
 
    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
 
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
 
            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
 
        if (keep)
            picked.push_back(i);
    }
}
 
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
 
// unsigmoid
static inline float unsigmoid(float y) {
    return static_cast<float>(-1.0 * (log((1.0 / y) - 1.0)));
}
 
static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, float prob_threshold,
                               std::vector <Object> &objects) {
    const int num_grid = feat_blob.h;
    float unsig_pro = 0;
    if (prob_threshold > 0.6)
        unsig_pro = unsigmoid(prob_threshold);
 
    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }
 
    const int num_class = feat_blob.w - 5;
 
    const int num_anchors = anchors.w / 2;
 
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
 
        const ncnn::Mat feat = feat_blob.channel(q);
 
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);
 
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                float box_score = featptr[4];
                if (prob_threshold > 0.6) {
                    // while prob_threshold > 0.6, unsigmoid better than sigmoid
                    if (box_score > unsig_pro) {
                        for (int k = 0; k < num_class; k++) {
                            float score = featptr[5 + k];
                            if (score > class_score) {
                                class_index = k;
                                class_score = score;
                            }
                        }
 
                        float confidence = sigmoid(box_score) * sigmoid(class_score);
 
                        if (confidence >= prob_threshold) {
 
                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);
 
                            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.f - 0.5f + i) * stride;
 
                            float pb_w = pow(dw * 2.f, 2) * anchor_w;
                            float pb_h = pow(dh * 2.f, 2) * anchor_h;
 
                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;
 
                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = confidence;
 
                            objects.push_back(obj);
                        }
                    } else {
                        for (int k = 0; k < num_class; k++) {
                            float score = featptr[5 + k];
                            if (score > class_score) {
                                class_index = k;
                                class_score = score;
                            }
                        }
                        float confidence = sigmoid(box_score) * sigmoid(class_score);
 
                        if (confidence >= prob_threshold) {
                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);
 
                            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.f - 0.5f + i) * stride;
 
                            float pb_w = pow(dw * 2.f, 2) * anchor_w;
                            float pb_h = pow(dh * 2.f, 2) * anchor_h;
 
                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;
 
                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = confidence;
                            objects.push_back(obj);
                        }
                    }
                }
            }
        }
    }
}
 
static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov5;
 
#if USE_INT8
    yolov5.opt.use_int8_inference=true;
#else
    yolov5.opt.use_vulkan_compute = true;
    yolov5.opt.use_bf16_storage = true;
#endif
 
    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
 
#if 1
    //yolov5.load_param("/home/aono/Downloads/ncnn/build/modles/3/eopt.param");
    //yolov5.load_model("/home/aono/Downloads/ncnn/build/modles/3/eopt.bin");
    yolov5.load_param("/home/aono/Downloads/ncnn/build/modles/5-1/eopt.param");
    yolov5.load_model("/home/aono/Downloads/ncnn/build/modles/5-1/eopt.bin");
#else
    yolov5.load_param("/home/aono/Downloads/ncnn/build/modles/5/e.pnnx.param");
    yolov5.load_model("/home/aono/Downloads/ncnn/build/modles/5/e.pnnx.bin");
    //yolov5.load_param("/home/aono/Documents/ncnn/build/eopt_8.param");
    //yolov5.load_model("/home/aono/Documents/ncnn/build/eopt_8.bin");
#endif
 
    const int target_size = 320;
    const float prob_threshold = 0.60f;
    const float nms_threshold = 0.60f;
 
    int img_w = bgr.cols;
    int img_h = bgr.rows;
 
    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
 
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
 
    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
 
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
 
    ncnn::Extractor ex = yolov5.create_extractor();
 
    ex.input("images", in_pad);
 
    std::vector<Object> proposals;
 
    // stride 8
    {
        ncnn::Mat out;
        ex.extract("579", out);
 
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;
 
        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
 
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    // stride 16
    {
        ncnn::Mat out;
        ex.extract("591", out);
 
 
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;
 
        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
 
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    // stride 32
    {
        ncnn::Mat out;
        ex.extract("603", out);
 
 
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;
 
        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
 
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
 
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
 
        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
 
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
 
    return 0;
}
 
struct result {
    int label;  // 物体代号
    int x,y;
};
 
struct CallbackParams {
	bool selectObject;
	Rect selection;
	Mat xyz;
	Mat grayImageL;
	Mat grayImageR;
	Mat rectifyImageL;
	Mat rectifyImageR;
	Mat Q;
	int blockSize=6;
	int n=5;
  int d;
  int y_bottom;
};

result draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects,void *data)
{
    CallbackParams* params = static_cast<CallbackParams*>(data);
    int center_x = 0;
    int center_y = 0;
    int center_label = -1;
    char name[32] = "";
    static const char* class_names[] = {
        "Cone barrel"
    };
 
    cv::Mat image = bgr.clone();
 
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
 
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                //obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
 
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0)); 
 
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
 
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
 
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
 
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
 
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); 
                    
        params->y_bottom = obj.rect.y;
        center_x = obj.rect.x + obj.rect.width / 2;
        center_y = obj.rect.y + obj.rect.height / 2;
        center_label = obj.label;
        cv::circle(image, cv::Point(center_x, center_y), 5, Scalar(255, 255, 255), -1);
        if (waitKey(30) == 27) break;
        if (cv::waitKey(30) >= 0) break;
    }
#if USE_CAMERA
    
    // only show left camera
    //int width = image.cols / 2;  
    //int height = image.rows;

    //cv::Mat left_half = image(cv::Rect(0, 0, width, height)).clone();

    //imshow("camera", left_half);
    //cv::waitKey(1);
    
    //original code
    imshow("camera", image);

    cv::waitKey(1);
#else
    cv::imwrite("result.jpg", image);
#endif

    result obj_info;
    obj_info.label = center_label;  // 确保字符串以 '\0' 结尾
    obj_info.x = center_x;
    obj_info.y = center_y;

    
    return obj_info;
}

using namespace std;
using namespace cv;



/**
 * @brief 计算物体连线相对于垂直地面的角度
 * 
 * @param y_bottom 物体底部在图像中的 y 坐标（像素）
 * @param img_height 图像总高度（像素）
 * @param fov_vertical 摄像头的垂直视场角（度）
 * @param camera_angle 摄像头光轴相对于地面的角度（度）
 * @return double 物体连线相对于垂直地面的角度（度）
 */
double computeObjectAngle(void* data) {
    CallbackParams* params = static_cast<CallbackParams*>(data);
    double img_height = HEIGHT;
    double fov_vertical = FOV_VERTICAL;
    double camera_angle = CAMERA_ANGLE;
    // 计算物体相对摄像头光轴的角度偏移
    double theta_offset = fov_vertical * (1.0 - (2.0 * params->y_bottom / img_height));

    // 计算物体连线相对于垂直地面的角度
    double alpha = camera_angle + theta_offset;

    return alpha;
}


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
    std::lock_guard<std::mutex> lock(data_mutex);

    distances = d;

    
    
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
	//cv::circle(disColorMap, cv::Point(x, y), 5, Scalar(255, 255, 255), -1);
	//imshow("depth", disColorMap);


}


// 原子变量控制线程运行
std::atomic<bool> keep_running(true);

// 定时发送线程函数
void serialThreadFunc() {

    //CallbackParams* params = static_cast<CallbackParams*>(data);
    //d = params->d;
    while (keep_running) {
        char buffer[32];
        //snprintf(buffer, sizeof(buffer), " %d \n", d);
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            snprintf(buffer, sizeof(buffer), " %.2f m\n", distances);
        }

        auto start = std::chrono::steady_clock::now();
        int serial_fd;
        serial_fd = serialOpen(SERIAL_DEVICE, BAUD_RATE);
        

        // 执行串口发送
        //serialPuts(serial_fd, buffer);
        if(distances<3){
           
          serialPuts(serial_fd, buffer);        
        }

        serialClose(serial_fd);
        // 计算精确的等待时间
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto wait_time = std::chrono::seconds(2) - elapsed;

        if (wait_time.count() > 0) {
            std::this_thread::sleep_for(wait_time);
        }
    }
}



#if USE_CAMERA

int main(int argc, char** argv){ 
    
    /************shared memory****************/
    // 创建共享内存
    int shmid = shmget(SHM_KEY, SHM_SIZE, 0666 | IPC_CREAT);
    if (shmid == -1) {
        std::cerr << "Failed to create shared memory\n";
        return 1;
    }

    uchar* shm_ptr = (uchar*)shmat(shmid, nullptr, 0); // 连接共享内存
    if (shm_ptr == (uchar*)-1) {
        std::cerr << "Failed to attach shared memory\n";
        return 1;
    }
    /************shared memory****************/  
    

    //open camera
  	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    VideoCapture cap(0);//打开摄像头（输入摄像头的索引）
    
  	//单个摄像头的分辨率  
  	const int imageWidth = 640;
  	const int imageHeight = 480;
  
  	//单个相机的分辨率
  	Size imageSize = Size(imageWidth, imageHeight);

  	cap.set(CAP_PROP_FRAME_WIDTH, imageWidth);
  	cap.set(CAP_PROP_FRAME_HEIGHT, imageHeight);
   
  	if (!cap.isOpened()) {
  	cout << "Error: Could not open the camera." << endl;
  	return -1;
  	}
    cv::Mat frame;   
          
  	CallbackParams params;
    while (true)
    {
        cap >> frame;
        cv::Mat m = frame;
        if (frame.empty()) break;

        std::vector<Object> objects;
        
        detect_yolov5(frame, objects);
        
        result obj = draw_objects(m, objects,&params);  
        
        
        //std::cout << "angle: " << tan(computeObjectAngle(&params)* M_PI / 180.0)/2 <<"\n";
        // 调整大小以匹配共享内存
        cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));

        // 复制数据到共享内存
        std::memcpy(shm_ptr, frame.data, SHM_SIZE);
        
        // 将检测数据存入共享内存（图像数据之后）
        std::memcpy(shm_ptr + IMG_SIZE, &obj, sizeof(result));

        std::cout << "Frame & Object written to shared memory: " << obj.label 
                  << " (" << obj.x << "," << obj.y << ")\n";
        std::cout << "Frame written to shared memory\n";
        usleep(300000);  // 每 30ms 采集一次（约 30 FPS）
        
                
        int key = cv::waitKey(1); // 监听键盘输入
        if (key == 27) { // ESC 键的 ASCII 码是 27
            std::cout << "ESC 键被按下，退出循环" << std::endl;
            break;
        }
    }

    shmdt(shm_ptr);  // 断开共享内存
    shmctl(shmid, IPC_RMID, nullptr);  // 删除共享内存
    cap.release();
    return 0;
}
#else
int main(int argc, char** argv)
{
    if (argc != 2)
	  {
	      fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
	      return -1;
	  }
	
	  const char* imagepath = argv[1];
        
	  struct timespec begin, end;
          long time;
          clock_gettime(CLOCK_MONOTONIC, &begin);
        
  cv::Mat m = cv::imread(imagepath, 1);
          if (m.empty())
          {
              fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
	
   std::vector<Object> objects;
   detect_yolov5(m, objects);
 
    clock_gettime(CLOCK_MONOTONIC, &end);
    time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
    printf(">> Time : %lf ms\n", (double)time/1000000);
 
    draw_objects(m, objects);
 
    return 0;
}
#endif

