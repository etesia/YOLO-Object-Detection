// 這份code要OpenCV 3.4.3以上才能用唷! 3.4.2不行!

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{image i        |<none>| input image   }" // 讀影像
"{video v        |<none>| input video   }" // 讀影片
"{device d       |0|    }" // 讀攝影機
;

using namespace cv;
using namespace dnn;
using namespace std;

// 初始化參數
float confThreshold = 0.5; // Confidence 閾值
float nmsThreshold = 0.4;  // NMS(Non-maximum suppression) 閾值
int inpWidth = 416;  // 輸入影像的寬
int inpHeight = 416; // 輸入影像的高
// 在Darknet中提供的config檔，有其他解析度供下載(解析度小速度快)，在這邊寬高設定需要跟config檔相同size
vector<string> classes; // 宣告vector 裡面存類別的名字 (在這個範例中，總共有80種類別，可參考coco.names這個檔案)

// NMS作後處理，移除重疊過高及太低分的Bounding boxes
void postprocess(Mat& frame, const vector<Mat>& out);

// 此函式用來畫出預測的Bounding boxes
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// 取得"輸出層"的名字 (即最後一層的名字)
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	// 讀入物件類別的名字，由coco.names取得，共80類
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// 讀入config, weights檔 (皆由原作者維護的的Darknet取得)
	String modelConfiguration = "yolov3.cfg"; // 內容是網路架構 (conv, pool, activation function...etc)
	String modelWeights = "yolov3.weights";  // 訓練好模型的參數權重
	
	// 將模型透過readNetFromDarknet讀進來
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV); // 使用Opencv backend
	net.setPreferableTarget(DNN_TARGET_OPENCL); 
	// 這邊代表可用GPU or CPU跑，如果你設定GPU，但你的GPU不支援，會自動轉成CPU
	// DNN_TARGET_OPENCL -> intel GPU
	// DNN_TARGET_CPU -> 當前電腦的CPU 



	// 用於影像or影片or攝影機前的初始宣告
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;


	///start
	try {

		outputFile = "yolo_out_cpp.mp4"; // 輸出的檔名
		if (parser.has("image")) // 第一個if 先看你是否有影像參數，有就進來做
		{
			//* 打開影像檔案 *//
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("error"); // 若找不到丟回error
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg"); // 存輸出影像
			outputFile = str;
		}
		else if (parser.has("video")) // 第二個if 如果你有影片參數 就進來做
		{
			//* 打開影片檔案 *//
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error"); // 若找不到丟回error
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
		}
		// 如果沒給影像or影片，就開的webcam
		else cap.open(parser.get<int>("device"));

	}
	
	catch (...) // catch(...) 代表"任何例外"
	{ 
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}
	
	// 用VideoWriter 儲存輸出的影像or影片
	if (!parser.has("image")) {
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}
	///end


	// 開一個視窗
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL); // WINDOW_NORMAL -> 可改視窗大小

	// 處理每一個frame
	while (waitKey(1) < 0)
	{
		// 讀入video的每一個frame
		cap >> frame;

		// 如果frame跑到空了 代表跑完了
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(3000);
			break;
		}
		// Blob (binary large object) 
		// 是四維的資料結構，把影像轉為 N*C*H*W (N: batch size, C: Channel, 高, 寬)
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		// 將Blob丟進網路
		net.setInput(blob);
		// 進行前向傳播取得輸出層的輸出
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));
		// NMS作後處理，移除重疊過高及太低分的Bounding boxes
		postprocess(frame, outs);

		// 這邊是計算整個網路的偵測速度，在畫面右上角用紅色顯示
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// 將預測的B-box寫入frame中
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (parser.has("image")) imwrite(outputFile, detectedFrame);
		else video.write(detectedFrame);

		imshow(kWinName, frame);

	}

	cap.release();
	if (!parser.has("image")) video.release();

	return 0;
}

// NMS作後處理，移除重疊過高及太低分的Bounding boxes
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// 掃描出的所有B-boxes，保留高分數的B-boxes，把類別標籤指定給最高分的B-box。
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			// 取得最高分Box的分數以及位置
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;
				/* 這邊data[0], data[1], data[2], data[3] 是"百分比"
				   所以必須乘以影像的寬、高，才會是作者論文中的 x, y, w, h
				   x, y: bounding box相對於grid cell的位移量
				   w, h: bounding box的寬、高
				*/

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// 使用NMS來消除一些過度重疊的B-boxes
	vector<int> indices;
	// NMSBoxes需要的參數依序為: Boxes, 信心值, 信心閾值, nms閾值, 保存boxes的indices
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

	//* 繪製預測的Bounding box *//
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	// 繪製矩形框出目標
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//取得並秀出類別名字 還有他的信心分數
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	// 在Bounding box上putText打上類別名字
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// 取得輸出層(最後一層)
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		// 找出輸出層，透過getUnconnectedOutLayers() 這個函式去找
		// 意思是如果他沒有連接outputs, 就代表此層是輸出層了
		vector<int> outLayers = net.getUnconnectedOutLayers();
		
		// 把網路中所有層的名字取出來 conv, pool......etc
		vector<String> layersNames = net.getLayerNames();

		// 取得最後一層的layer名稱
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
		/* 
		此時 names 裡面會有3個string 如果印出來看會是 "yolo_82", "yolo_94", "yolo_106" 這三個為輸出層
		照理說輸出層抓出來應該只有一個，但這邊是yolov3採取FPN架構(增強小物件偵測)，所以會有三個
		可參閱 feature pyramid networks (FPN) 論文 
		*/
	}
	return names;
}