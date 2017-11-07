#include "fp_detector.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int FPDetector::FindFaceLandMark(const cv::Mat & img)
{
  facial_points.clear();
  img.convertTo(blob_image_, CV_32FC1);
  cv::resize(blob_image_, blob_image_, dsize_, 0, 0, cv::INTER_CUBIC);
  cv::Mat tmp_m, tmp_sd;
  meanStdDev(blob_image_, tmp_m, tmp_sd);
  double mean = tmp_m.at<double>(0, 0);
  double std_dev = tmp_sd.at<double>(0, 0);
  blob_image_ = (blob_image_ - mean) / (0.000001 + std_dev);

  inputBlob_ = cv::dnn::blobFromImage(blob_image_);
  net_.setInput(inputBlob_, "data");
  outBlob_ = net_.forward("Dense3");
  const float* data_ptr = (const float*)outBlob_.data;
  cv::Point2f temp_point;
  for (size_t i = 0; i < num_points_*2; i += 2) {
    temp_point.x = *(data_ptr + i);
    temp_point.y = *(data_ptr + i + 1);
    facial_points.push_back(temp_point);
  }

  return 0;
}

int FPDetector::LoadModel(const std::string & config_path, const std::string & model_path)
{
  net_ = cv::dnn::readNetFromCaffe(config_path, model_path);
  return 0;
}
