#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>

#include "simple_timer.hpp"

namespace fs = boost::filesystem;

std::vector<fs::path> GetAllImagePaths(fs::path const &root) {
  std::vector<fs::path> image_paths;
  if (fs::exists(root) && fs::is_directory(root)) {
    for (auto const &entry : fs::recursive_directory_iterator(root)) {
      if (fs::is_regular_file(entry) && (entry.path().extension() == ".png" ||
                                         entry.path().extension() == ".jpg" ||
                                         entry.path().extension() == ".jpeg")) {
        image_paths.push_back(entry.path());
      }
    }
  }
  return image_paths;
}

cv::Mat loadFeature(const std::string &filepath,
                    const std::string &descriptor) {
  cv::Ptr<cv::Feature2D> fdetector;
  if (descriptor == "orb")
    fdetector = cv::ORB::create();
  else if (descriptor == "brisk")
    fdetector = cv::BRISK::create();
  else
    throw std::runtime_error("Invalid descriptor");
  assert(!descriptor.empty());

  cv::Mat image = cv::imread(filepath, 0);
  if (image.empty())
    throw std::runtime_error("Could not open image" + filepath);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
  return descriptors;
}

std::vector<cv::Mat> loadFeatures(const std::vector<fs::path> &path_to_images,
                                  const std::string &descriptor = "") {
  // select detector
  cv::Ptr<cv::Feature2D> fdetector;
  if (descriptor == "orb")
    fdetector = cv::ORB::create();
  else if (descriptor == "brisk")
    fdetector = cv::BRISK::create();
  else
    throw std::runtime_error("Invalid descriptor");
  assert(!descriptor.empty());
  std::vector<cv::Mat> features;

  std::cout << "Extracting   features..." << std::endl;
  for (size_t i = 0; i < path_to_images.size(); ++i) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // std::cout << "reading image: " << path_to_images[i] << std::endl;
    cv::Mat image = cv::imread(path_to_images[i].string().c_str(), 0);
    if (image.empty())
      throw std::runtime_error("Could not open image" +
                               path_to_images[i].string());
    // std::cout << "extracting features" << std::endl;
    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    // std::cout << "done detecting features" << std::endl;
  }
  return features;
}

void PrintUsage(char *argv[]) {
  std::cout << "Usage: " << argv[0]
            << " <voc_path>  <image_to_create_database_dir> <query_image> "
               "<feature_name>"
            << std::endl;
  std::cout << "  feature_name is:" << std::endl;
  std::cout << "     orb" << std::endl;
  std::cout << "     brisk" << std::endl;
  std::exit(0);
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    PrintUsage(argv);
  }

  std::string feature_name = argv[4];
  std::string query_image = argv[3];
  std::string image_dir = argv[2];
  std::string voc_path = argv[1];

  std::cout << "Vocabulary path: " << voc_path << std::endl;
  std::cout << "Feature name: " << feature_name << std::endl;
  std::cout << "Image dir: " << image_dir << std::endl;

  std::vector<fs::path> image_paths = GetAllImagePaths(image_dir);
  std::cout << "Number of images: " << image_paths.size() << std::endl;

  StopWatchTimer feat_timer{"load featrues"};
  feat_timer.start();
  auto features = loadFeatures(image_paths, feature_name);
  feat_timer.stop();

  auto query_features = loadFeature(query_image, feature_name);

  DBoW3::Vocabulary voc(voc_path);

  // create a database
  StopWatchTimer db_timer{"create db"};
  db_timer.start();
  DBoW3::Database db(voc, false, 0);
  for (auto const &feat : features) {
    db.add(feat);
  }
  db_timer.stop();

  DBoW3::QueryResults ret;
  StopWatchTimer query_timer{"query"};
  query_timer.start();
  db.query(query_features, ret, 4);
  query_timer.stop();
  std::cout << "Quering for Image: " << query_image << ": " << ret << std::endl;

  feat_timer.print_stats();
  db_timer.print_stats();
  query_timer.print_stats();
}
