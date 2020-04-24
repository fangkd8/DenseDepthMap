#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <limits>
#include <chrono>
#include <omp.h>
#include <array>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
namespace fs = std::experimental::filesystem;
namespace ba = boost::algorithm;

string LIDAR_FILE = "";

class projection{
public:
  projection(){
    Eigen::initParallel();
    Eigen::setNbThreads(12);
    MatrixXf data = readbinfile(LIDAR_FILE);

    MatrixXf P;
    P.resize(3, 4);
    P << 609.6954, -721.4216, -1.2513,   -123.0418,
         180.3842,  7.6448,   -719.6515, -101.0167,
         0.9999,    1.2437e-4, 0.0105,   -0.2694;
    auto start = high_resolution_clock::now();  

    data = points_filter(P, data); 
    cv::Mat result = DenseMap(data, 4);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << " microseconds" << endl;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    result = 255 * (result - minVal) / (maxVal - minVal);
    result.convertTo(result, CV_8UC1);
    cv::imshow("result", result);
    cv::waitKey(0);
  }

  MatrixXf readbinfile(const string dir){

    ifstream fin(dir.c_str(), ios::binary);
    assert(fin);
  
    fin.seekg(0, ios::end);
    const size_t num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, ios::beg);
  
    vector<float> l_data(num_elements);
    fin.read(reinterpret_cast<char*>(&l_data[0]), num_elements*sizeof(float));
  
    MatrixXf data = Map<MatrixXf>(l_data.data(), 4, l_data.size()/4);
  
    return data;
  }

  MatrixXf points_filter(MatrixXf &P, MatrixXf &data){
    data = P * data;
    vector<int> v;
    omp_set_num_threads(16);
    #pragma omp parallel
    {
      vector<int> v1;
      #pragma omp for nowait
      for (int j = 0; j < data.cols(); j++){
        if (data(2, j) > 0){
          float x = data(0, j) / data(2, j);
          float y = data(1, j) / data(2, j);
          if ( (x > 0 && x < COL - 0.5) && (y > 0 && y < ROW - 0.5) )
            v1.push_back(j);
        }
      }
      #pragma omp critical
      v.insert(v.end(), v1.begin(), v1.end());
    }
    MatrixXf result;
    result.resize(3, v.size());
    result.fill(0.);
    cout << data.rows() << " " << data.cols() << endl;
    omp_set_num_threads(16);
    #pragma omp parallel
    { 
      MatrixXf res_private;
      res_private.resize(3, v.size());
      res_private.fill(0.);
      #pragma omp for nowait
      for (auto i = 0; i < v.size(); i++){
        res_private(0, i) = data(0, v[i]) / data(2, v[i]);
        res_private(1, i) = data(1, v[i]) / data(2, v[i]);
        res_private(2, i) = data(2, v[i]);

      }
      #pragma omp critical
      result += res_private;
    }
    return result;
  }

  cv::Mat DenseMap(MatrixXf &data, int grid){
    int ng = 2 * grid + 1;

    cv::Mat map, mD;
    map = cv::Mat::zeros(ROW, COL, CV_32FC1);
    mD = cv::Mat::zeros(ROW, COL, CV_32FC1);
    omp_set_num_threads(8);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for (auto i = 0; i < data.cols(); i++){
        map.at<float>(round(data(1, i)), round(data(0, i))) = 
          sqrt(pow(data(0, i) - round(data(0, i)), 2) + 
               pow(data(1, i) - round(data(1, i)), 2));
        mD.at<float>(round(data(1, i)), round(data(0, i))) = data(2, i);
      }
    }

    cv::Mat output;
    output = cv::Mat::zeros(ROW, COL, CV_32FC1);
    omp_set_num_threads(128);
    #pragma omp parallel
    {
      #pragma omp for nowait
      for (auto i = 0; i < ROW; i++){
        for (auto j = 0; j < COL; j++){
          if (i - grid < 0 || i + grid >= ROW)
            continue;
          if (j - grid < 0 || j + grid >= COL) 
            continue;
          float s = 0;
          for (auto r = -grid; r < grid + 1; r++){
            for (auto c = -grid; c < grid + 1; c++){
              float map_val = map.at<float>(i+r, j+c);
              if (map_val != 0){
                output.at<float>(i, j) += 
                  mD.at<float>(i+r, j+c) / map.at<float>(i+r, j+c);
                s += 1 / map.at<float>(i+r, j+c);
              }
            }
          }
          if (s == 0){
            s = 1;
          }
          output.at<float>(i, j) /= s;
        }
      }
    }
    return output;
  }

private:
  const int ROW = 375;
  const int COL = 1242;

};

int main(){
  projection p;
}