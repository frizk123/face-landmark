#ifndef FP_DETECTOR_H_
#define FP_DETECTOR_H_
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>

class FPDetector {
public:
  FPDetector() = default;
  ~FPDetector() = default;
  int FindFaceLandMark(const cv::Mat& img);
  int LoadModel(const std::string& config_path,
                const std::string& model_path);
  std::vector<cv::Point2f> facial_points;

private:
  cv::dnn::Net net_;
  cv::Size dsize_ = cv::Size(60, 60);
  cv::Mat inputBlob_;
  cv::Mat blob_image_;
  cv::Mat outBlob_;
  size_t num_points_ = 68;
};

#endif // !FP_DETECTOR_H_
