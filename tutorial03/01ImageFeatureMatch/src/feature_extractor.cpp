#include "feature_extractor.h"

FeatureExtractor::FeatureExtractor(const std::string &name)
    : name_(name), timer_(std::make_unique<StopWatchTimer>(name)) {}

FeatureExtractor::~FeatureExtractor() {
  name_ = "";
  timer_.reset(nullptr);
}

void FeatureExtractor::detect(const cv::Mat &image,
                              std::vector<cv::KeyPoint> &keypoints) {}

double FeatureExtractor::get_detect_time() { return timer_->elapsed_time_ms(); }

std::string FeatureExtractor::get_name() { return name_; }

SiftFeatureExtractor::SiftFeatureExtractor()
    : FeatureExtractor("SIFT"), sift_ptr_(cv::SIFT::create()) {}

SiftFeatureExtractor::~SiftFeatureExtractor() {
  FeatureExtractor::~FeatureExtractor();
  sift_ptr_ = nullptr;
}

void SiftFeatureExtractor::detect(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &keypoints) {
  FeatureExtractor::timer_->reset();
  FeatureExtractor::timer_->start();
  sift_ptr_->detect(image, keypoints);
  FeatureExtractor::timer_->stop();
}

SurfFeatureExtractor::SurfFeatureExtractor()
    : FeatureExtractor("SURF"), surf_ptr_(cv::xfeatures2d::SURF::create()) {}

SurfFeatureExtractor::~SurfFeatureExtractor() {
  FeatureExtractor::~FeatureExtractor();
  surf_ptr_ = nullptr;
}

void SurfFeatureExtractor::detect(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &keypoints) {
  FeatureExtractor::timer_->start();
  surf_ptr_->detect(image, keypoints);
  FeatureExtractor::timer_->stop();
}

FastFeatureExtractor::FastFeatureExtractor()
    : FeatureExtractor("FAST"), fast_ptr_(cv::ORB::create()) {}

FastFeatureExtractor::~FastFeatureExtractor() {
  FeatureExtractor::~FeatureExtractor();
  fast_ptr_ = nullptr;
}

void FastFeatureExtractor::detect(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &keypoints) {
  FeatureExtractor::timer_->start();
  fast_ptr_->detect(image, keypoints);
  FeatureExtractor::timer_->stop();
}

FeatureExtractor *create_extractor(const ExtractorType &type) {
  switch (type) {
  case ExtractorType::SIFT:
    return reinterpret_cast<FeatureExtractor *>(new SiftFeatureExtractor());
  case ExtractorType::SURF:
    return reinterpret_cast<FeatureExtractor *>(new SurfFeatureExtractor());
  case ExtractorType::FAST:
    return reinterpret_cast<FeatureExtractor *>(new FastFeatureExtractor());
  default:
    return reinterpret_cast<FeatureExtractor *>(new SiftFeatureExtractor());
  }
}
