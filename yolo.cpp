// �o��code�nOpenCV 3.4.3�H�W�~��έ�! 3.4.2����!

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{image i        |<none>| input image   }" // Ū�v��
"{video v        |<none>| input video   }" // Ū�v��
"{device d       |0|    }" // Ū��v��
;

using namespace cv;
using namespace dnn;
using namespace std;

// ��l�ưѼ�
float confThreshold = 0.5; // Confidence �H��
float nmsThreshold = 0.4;  // NMS(Non-maximum suppression) �H��
int inpWidth = 416;  // ��J�v�����e
int inpHeight = 416; // ��J�v������
// �bDarknet�����Ѫ�config�ɡA����L�ѪR�רѤU��(�ѪR�פp�t�ק�)�A�b�o��e���]�w�ݭn��config�ɬۦPsize
vector<string> classes; // �ŧivector �̭��s���O���W�r (�b�o�ӽd�Ҥ��A�`�@��80�����O�A�i�Ѧ�coco.names�o���ɮ�)

// NMS�@��B�z�A�������|�L���ΤӧC����Bounding boxes
void postprocess(Mat& frame, const vector<Mat>& out);

// ���禡�Ψӵe�X�w����Bounding boxes
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// ���o"��X�h"���W�r (�Y�̫�@�h���W�r)
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	// Ū�J�������O���W�r�A��coco.names���o�A�@80��
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Ū�Jconfig, weights�� (�ҥѭ�@�̺��@����Darknet���o)
	String modelConfiguration = "yolov3.cfg"; // ���e�O�����[�c (conv, pool, activation function...etc)
	String modelWeights = "yolov3.weights";  // �V�m�n�ҫ����Ѽ��v��
	
	// �N�ҫ��z�LreadNetFromDarknetŪ�i��
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV); // �ϥ�Opencv backend
	net.setPreferableTarget(DNN_TARGET_OPENCL); 
	// �o��N��i��GPU or CPU�]�A�p�G�A�]�wGPU�A���A��GPU���䴩�A�|�۰��নCPU
	// DNN_TARGET_OPENCL -> intel GPU
	// DNN_TARGET_CPU -> ��e�q����CPU 



	// �Ω�v��or�v��or��v���e����l�ŧi
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;


	///start
	try {

		outputFile = "yolo_out_cpp.mp4"; // ��X���ɦW
		if (parser.has("image")) // �Ĥ@��if ���ݧA�O�_���v���ѼơA���N�i�Ӱ�
		{
			//* ���}�v���ɮ� *//
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("error"); // �Y�䤣���^error
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg"); // �s��X�v��
			outputFile = str;
		}
		else if (parser.has("video")) // �ĤG��if �p�G�A���v���Ѽ� �N�i�Ӱ�
		{
			//* ���}�v���ɮ� *//
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error"); // �Y�䤣���^error
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
		}
		// �p�G�S���v��or�v���A�N�}��webcam
		else cap.open(parser.get<int>("device"));

	}
	
	catch (...) // catch(...) �N��"����ҥ~"
	{ 
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}
	
	// ��VideoWriter �x�s��X���v��or�v��
	if (!parser.has("image")) {
		video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}
	///end


	// �}�@�ӵ���
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL); // WINDOW_NORMAL -> �i������j�p

	// �B�z�C�@��frame
	while (waitKey(1) < 0)
	{
		// Ū�Jvideo���C�@��frame
		cap >> frame;

		// �p�Gframe�]��ŤF �N��]���F
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(3000);
			break;
		}
		// Blob (binary large object) 
		// �O�|������Ƶ��c�A��v���ର N*C*H*W (N: batch size, C: Channel, ��, �e)
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		// �NBlob��i����
		net.setInput(blob);
		// �i��e�V�Ǽ����o��X�h����X
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));
		// NMS�@��B�z�A�������|�L���ΤӧC����Bounding boxes
		postprocess(frame, outs);

		// �o��O�p���Ӻ����������t�סA�b�e���k�W���ά������
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// �N�w����B-box�g�Jframe��
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

// NMS�@��B�z�A�������|�L���ΤӧC����Bounding boxes
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// ���y�X���Ҧ�B-boxes�A�O�d�����ƪ�B-boxes�A�����O���ҫ��w���̰�����B-box�C
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			// ���o�̰���Box�����ƥH�Φ�m
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;
				/* �o��data[0], data[1], data[2], data[3] �O"�ʤ���"
				   �ҥH�������H�v�����e�B���A�~�|�O�@�̽פ夤�� x, y, w, h
				   x, y: bounding box�۹��grid cell���첾�q
				   w, h: bounding box���e�B��
				*/

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// �ϥ�NMS�Ӯ����@�ǹL�׭��|��B-boxes
	vector<int> indices;
	// NMSBoxes�ݭn���Ѽƨ̧Ǭ�: Boxes, �H�߭�, �H���H��, nms�H��, �O�sboxes��indices
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

	//* ø�s�w����Bounding box *//
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	// ø�s�x�ήإX�ؼ�
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//���o�èq�X���O�W�r �٦��L���H�ߤ���
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	// �bBounding box�WputText���W���O�W�r
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// ���o��X�h(�̫�@�h)
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		// ��X��X�h�A�z�LgetUnconnectedOutLayers() �o�Ө禡�h��
		// �N��O�p�G�L�S���s��outputs, �N�N���h�O��X�h�F
		vector<int> outLayers = net.getUnconnectedOutLayers();
		
		// ��������Ҧ��h���W�r���X�� conv, pool......etc
		vector<String> layersNames = net.getLayerNames();

		// ���o�̫�@�h��layer�W��
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
		/* 
		���� names �̭��|��3��string �p�G�L�X�Ӭݷ|�O "yolo_82", "yolo_94", "yolo_106" �o�T�Ӭ���X�h
		�Ӳz����X�h��X�����ӥu���@�ӡA���o��Oyolov3�Ĩ�FPN�[�c(�W�j�p���󰻴�)�A�ҥH�|���T��
		�i�Ѿ\ feature pyramid networks (FPN) �פ� 
		*/
	}
	return names;
}