#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "simple_timer.hpp"

class FeatureExtractor {
public:
  FeatureExtractor() = default;
  FeatureExtractor(const std::string &name);
  virtual ~FeatureExtractor();
  // FeatureExtractor(const FeatureExtractor&) = delete;
  // FeatureExtractor(const FeatureExtractor&&) = delete;

  virtual double get_detect_time();
  virtual std::string get_name();

  virtual void detect(const cv::Mat &image,
                      std::vector<cv::KeyPoint> &keypoints);

protected:
  std::string name_;
  std::unique_ptr<StopWatchTimer> timer_;
};

class SiftFeatureExtractor : FeatureExtractor {
public:
  SiftFeatureExtractor();
  ~SiftFeatureExtractor();
  void detect(const cv::Mat &image,
              std::vector<cv::KeyPoint> &keypoints) override;

public:
  cv::Ptr<cv::SIFT> sift_ptr_;
};

class SurfFeatureExtractor : FeatureExtractor {
public:
  SurfFeatureExtractor();
  ~SurfFeatureExtractor();
  void detect(const cv::Mat &image,
              std::vector<cv::KeyPoint> &keypoints) override;

public:
  cv::Ptr<cv::xfeatures2d::SURF> surf_ptr_;
};

class FastFeatureExtractor : FeatureExtractor {
public:
  FastFeatureExtractor();
  ~FastFeatureExtractor();
  void detect(const cv::Mat &image,
              std::vector<cv::KeyPoint> &keypoints) override;

public:
  cv::Ptr<cv::FeatureDetector> fast_ptr_;
};

enum class ExtractorType {
  SIFT = 0,
  SURF = 1,
  FAST = 2,
};

FeatureExtractor *create_extractor(const ExtractorType &type);
