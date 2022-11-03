#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

// join two path
std::string path_join(const std::string &base_path,
                      const std::string &sub_path) {
  if (base_path.back() == '/') {
    return base_path + sub_path;
  } else {
    return base_path + '/' + sub_path;
  }
}

void PrintHelp(char *argv[]) {
  std::cout << "Usage: " << argv[0]
            << " <bag_path> <output_dir> <image_prefix> <topic_name>";
}

int main(int argc, char *argv[]) {

  // use ros param to get image
  std::string bag_path;
  std::string output_path;
  std::string image_prefix;
  std::string topic;

  if (argc < 5) {
    PrintHelp(argv);
    return 1;
  }

  bag_path = argv[1];
  output_path = argv[2];
  image_prefix = argv[3];
  topic = argv[4];

  std::cout << "bag_path: " << bag_path << std::endl;
  std::cout << "output_path: " << output_path << std::endl;
  std::cout << "image_prefix: " << image_prefix << std::endl;
  std::cout << "topic: " << topic << std::endl;

  // open the rosbag
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);

  // set the topic
  std::vector<std::string> topics;
  topics.push_back(topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  cv_bridge::CvImagePtr cv_ptr;

  // extract the images
  int i = 0;
  for (rosbag::MessageInstance const m : view) {
    sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
    if (image != nullptr) {
      // std::cout << i << " : " << image->header.stamp << std::endl;

      cv_ptr =
          cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_8UC3);

      std::stringstream ss;
      ss << image_prefix << std::setw(5) << std::setfill('0') << i << ".png";
      std::string image_path = path_join(output_path, ss.str());
      cv::imwrite(image_path, cv_ptr->image);
      std::cout << "Saved " << image_path << std::endl;
      i += 1;
    }
  }

  bag.close();
}
