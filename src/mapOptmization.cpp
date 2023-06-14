#include "utility.h"

#include "sc_lio_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include "Scancontext.h"

using namespace gtsam;

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
  using namespace gtsam;

  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), std::fstream::out);

  for (const auto& key_value : _estimates)
  {
    auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
    if (!p)
      continue;

    const Pose3& pose = p->value();

    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1); // Point3
    auto col2 = R.column(2); // Point3
    auto col3 = R.column(3); // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
      << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
      << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
  }
}

class mapOptimization : public ParamServer
{

public:
  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values optimizedEstimate;
  ISAM2* isam;
  Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubLaserOdometryGlobal;
  ros::Publisher pubLaserOdometryIncremental;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubPath;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames;
  ros::Publisher pubRecentKeyFrames;
  ros::Publisher pubRecentKeyFrame;
  ros::Publisher pubCloudRegisteredRaw;
  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subCloud;
  ros::Subscriber subLoop;

  sc_lio_sam::cloud_info cloudInfo;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D; // giseop
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr laserCloudRaw;   // giseop
  pcl::PointCloud<PointType>::Ptr laserCloudRawDS; // giseop
  double laserCloudRawTime;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // downsampled surf featuer set from odoOptimization

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterSC; // giseop
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

  ros::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;
  std::mutex mtxLoopInfo;

  bool isDegenerate = false;
  Eigen::Matrix<float, 6, 6> matP;

  int laserCloudCornerFromMapDSNum = 0;
  int laserCloudSurfFromMapDSNum = 0;
  int laserCloudCornerLastDSNum = 0;
  int laserCloudSurfLastDSNum = 0;

  bool aLoopIsClosed = false;
  // map<int, int> loopIndexContainer; // from new to old
  multimap<int, int> loopIndexContainer; // from new to old // giseop

  vector<pair<int, int>> loopIndexQueue;
  vector<gtsam::Pose3> loopPoseQueue;
  // vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue; // Diagonal <- Gausssian <- Base
  vector<gtsam::SharedNoiseModel> loopNoiseQueue; // giseop for polymorhpisam (Diagonal <- Gausssian <- Base)

  deque<std_msgs::Float64MultiArray> loopInfoVec;

  nav_msgs::Path globalPath;

  Eigen::Affine3f transPointAssociateToMap;
  Eigen::Affine3f incrementalOdometryAffineFront;
  Eigen::Affine3f incrementalOdometryAffineBack;

  // // loop detector
  SCManager scManager;

  // data saver
  std::fstream pgSaveStream;     // pg: pose-graph
  std::fstream pgTimeSaveStream; // pg: pose-graph
  std::vector<std::string> edges_str;
  std::vector<std::string> vertices_str;

  // @yan
  std::string saveSCDDirectory;
  std::string saveCornerScanDirectory;
  std::string saveSurfScanDirectory;
  std::ofstream keyPoseStream;
  pcl::PointCloud<PointType>::Ptr laserCloudDense;
  pcl::PointCloud<PointType>::Ptr laserCloudDenseDS;
  pcl::VoxelGrid<PointType> downSizeFilterDense;
  vector<pcl::PointCloud<PointType>::Ptr> denseCloudKeyFrames;
  std::ofstream poseStream;
  ros::Publisher pubDirCreated;
  int keyPoseNum = 0;


public:
  mapOptimization()
  {
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    // publisher & subscriber
    pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/trajectory", 1);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/map_global", 1);
    pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>("sc_lio_sam/mapping/odometry", 1);
    pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("sc_lio_sam/mapping/odometry_incremental", 1);
    pubPath = nh.advertise<nav_msgs::Path>("sc_lio_sam/mapping/path", 1);

    subCloud = nh.subscribe<sc_lio_sam::cloud_info>("sc_lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
    // 接受外部回环检测器的回环消息，配合 detectLoopClosureExternal 使用，暂未使用
    subLoop = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/sc_lio_sam/mapping/loop_closure_constraints", 1);

    pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/map_local", 1);
    pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("sc_lio_sam/mapping/cloud_registered_raw", 1);

    // downSizeFilter
    downSizeFilterSC.setLeafSize(mappingSCLeafSize, mappingSCLeafSize, mappingSCLeafSize);
    downSizeFilterICP.setLeafSize(mappingICPLeafSize, mappingICPLeafSize, mappingICPLeafSize);
    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
    downSizeFilterDense.setLeafSize(mappingDenseLeafSize, mappingDenseLeafSize, mappingDenseLeafSize);

    allocateMemory();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    // create directory and remove old files;
    if (exists(mapDirectory))
      system((std::string("exec rm -r ") + mapDirectory).c_str());

    saveSCDDirectory = mapDirectory + "SCDs/"; // SCD: scan context descriptor
    saveCornerScanDirectory = mapDirectory + "CornerScans/";
    saveSurfScanDirectory = mapDirectory + "SurfScans/";

    system((std::string("mkdir ") + mapDirectory).c_str());
    system((std::string("mkdir -p ") + saveSCDDirectory).c_str());
    system((std::string("mkdir -p ") + saveCornerScanDirectory).c_str());
    system((std::string("mkdir -p ") + saveSurfScanDirectory).c_str());

    pgSaveStream = std::fstream(mapDirectory + "singlesession_posegraph.g2o", std::fstream::out);
    pgTimeSaveStream = std::fstream(mapDirectory + "times.txt", std::fstream::out);
    pgTimeSaveStream.precision(dbl::max_digits10);

    poseStream.open(mapDirectory + "lidarOdometry.txt");

    // tell imuPreintegration that map directory is created
    pubDirCreated = nh.advertise<std_msgs::Bool>("sc_lio_sam/mapping/directory_created", 1);
    std_msgs::Bool dirCreated;
    dirCreated.data = true;
    while (pubDirCreated.getNumSubscribers() == 0)
    {
      ;
    }
    pubDirCreated.publish(dirCreated);
  }

  void allocateMemory()
  {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudRaw.reset(new pcl::PointCloud<PointType>());   // giseop
    laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // giseop

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   // corner feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     // surf feature set from odoOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled surf featuer set from odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
      transformTobeMapped[i] = 0;
    }

    matP.setZero();

    // @yan
    laserCloudDense.reset(new pcl::PointCloud<PointType>());
    laserCloudDenseDS.reset(new pcl::PointCloud<PointType>());
  }

  /*************** 构造函数结束 ***************/

  /*****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************/

   /*************** lidar scan 回调函数开始 ***************/

  void laserCloudInfoHandler(const sc_lio_sam::cloud_infoConstPtr& msgIn)
  {

    // extract time stamp
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur = msgIn->header.stamp.toSec();

    // extract info and feature cloud
    cloudInfo = *msgIn;

    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
    pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw); // giseop
    pcl::fromROSMsg(msgIn->cloud_dense, *laserCloudDense);
    laserCloudRawTime = cloudInfo.header.stamp.toSec();     // giseop save node time

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1;
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
    {
      timeLastProcessing = timeLaserInfoCur;

      updateInitialGuess();

      extractSurroundingKeyFrames();

      downsampleCurrentScan();

      scan2MapOptimization();

      saveKeyFramesAndFactor();

      correctPoses();

      publishOdometry();

      publishFrames();
    }
  }

  void updateInitialGuess()
  {
    /* 本函数的核心目的就是给 transformTobeMapped 赋初值 */

    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastImuTransformation;
    // initialization
    if (cloudKeyPoses3D->points.empty())
    {
      transformTobeMapped[0] = cloudInfo.imuRollInit;
      transformTobeMapped[1] = cloudInfo.imuPitchInit;
      // transformTobeMapped[2] = cloudInfo.imuYawInit;
      transformTobeMapped[2] = 0; // 原代码的意思是如果不用 gps 的话，就不用 imu 的 yaw 值作为初始值，直接用 0 作为初始值

      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
      return;
    }

    // use imu pre-integration estimation for pose guess
    // 优先用 imu 里程计提供的位姿作为初始估计位姿
    static bool lastImuPreTransAvailable = false;
    static Eigen::Affine3f lastImuPreTransformation;
    if (cloudInfo.odomAvailable == true)
    {
      // 从 imu 里程计获取到的当前帧位姿
      Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
        cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
      if (lastImuPreTransAvailable == false)
      {
        lastImuPreTransformation = transBack;
        lastImuPreTransAvailable = true;
      }
      else
      {
        Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        lastImuPreTransformation = transBack;
        // 在使用 imu 里程计信息时还一直更新 lastImuTransformation 是为了防止 imu 里程计不可用时使用 imu 的磁力计数据获取当前帧的旋转位姿
        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
        return;
      }
    }

    // use imu incremental estimation for pose guess (only rotation)
    // 如果 imu 里程计不可用，那么就用 imu 的磁力计数据获取当前帧的旋转位姿
    if (cloudInfo.imuAvailable == true)
    {
      // 从 imu 数据获取到的当前帧位姿（仅包含旋转）
      Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
      Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

      Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
      return;
    }
  }

  void extractSurroundingKeyFrames()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;

    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
    {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // also extract some latest key frames in case the robot rotates in one position
    // 防止机器人在一个位置旋转，从而不光基于距离来提取 local map，还讲最近 10 s 内的关键帧也加入到 local map 中
    // 存在一个问题：根据时间筛选的关键帧和根据距离筛选的关键帧可能会有重复，代码中并没有去重
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
      if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
      else
        break;
    }

    extractCloud(surroundingKeyPosesDS);
  }

  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
  {
    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
      // 这一步实际上是排除最近 10 s 内的那些距离超过 surroundingKeyframeSearchRadius 的关键帧
      if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
        continue;

      int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
      {
        // transformed cloud available
        // 如果 laserCloudMapContainer 里有这个键值对，就直接分别把 value 里的边缘点和平面点加到相应的 local map 中
        *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
        *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
      }
      else
      {
        // transformed cloud not available
        // 如果 laserCloudMapContainer 里没有这个键值对，就把这个关键帧的边缘点和平面点变换到当前帧坐标系下，然后加到相应的 local map 中，并制作这个键值对
        pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *laserCloudCornerFromMap += laserCloudCornerTemp;
        *laserCloudSurfFromMap += laserCloudSurfTemp;
        laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
      }
    }

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
      laserCloudMapContainer.clear();
  }

  void downsampleCurrentScan()
  {
    // giseop
    laserCloudRawDS->clear();
    downSizeFilterSC.setInputCloud(laserCloudRaw);
    downSizeFilterSC.filter(*laserCloudRawDS);

    // dense cloud
    // pcl::PointCloud<PointType>::Ptr frontDenseCloud; // 前方 180° 的点
    // frontDenseCloud.reset(new pcl::PointCloud<PointType>());

    // for (int rowIdx = 0; rowIdx < N_SCAN; rowIdx++)
    //   for (int colIdx = Horizon_SCAN / 4, iter = 0; iter < Horizon_SCAN / 2; iter++, colIdx = (colIdx + 1) % Horizon_SCAN)
    //     if (laserCloudDense->points[colIdx + rowIdx * Horizon_SCAN].intensity != -1)
    //       frontDenseCloud->push_back(laserCloudDense->points[colIdx + rowIdx * Horizon_SCAN]);

    laserCloudDenseDS->clear();
    downSizeFilterDense.setInputCloud(laserCloudDense);
    downSizeFilterDense.filter(*laserCloudDenseDS);

    // Downsample cloud from current scan
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
  }

  void scan2MapOptimization()
  {
    if (cloudKeyPoses3D->points.empty())
      return;

    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    {
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

      for (int iterCount = 0; iterCount < 30; iterCount++)
      {
        laserCloudOri->clear();
        coeffSel->clear();

        cornerOptimization();
        surfOptimization();

        combineOptimizationCoeffs();

        if (LMOptimization(iterCount) == true)
          break;
      }

      transformUpdate(); // 结合 imu 信息更新 transformTobeMapped（主要是用 imu 提供的 roll 和 pitch 角）
    }
    else
    {
      ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
  }

  void cornerOptimization()
  {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudCornerLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      if (pointSearchSqDis[4] < 1.0)
      {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++)
        {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++)
        {
          float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        cv::eigen(matA1, matD1, matV1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
        {

          float x0 = pointSel.x;
          float y0 = pointSel.y;
          float z0 = pointSel.z;
          float x1 = cx + 0.1 * matV1.at<float>(0, 0);
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0);
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);

          float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

          float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

          float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          float ld2 = a012 / l12;

          float s = 1 - 0.9 * fabs(ld2);

          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1)
          {
            laserCloudOriCornerVec[i] = pointOri;
            coeffSelCornerVec[i] = coeff;
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }
  }

  void surfOptimization()
  {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudSurfLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0;
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f matX0;

      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();

      if (pointSearchSqDis[4] < 1.0)
      {
        for (int j = 0; j < 5; j++)
        {
          matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }

        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++)
        {
          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
            pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
            pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
          {
            planeValid = false;
            break;
          }
        }

        if (planeValid)
        {
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

          coeff.x = s * pa;
          coeff.y = s * pb;
          coeff.z = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1)
          {
            laserCloudOriSurfVec[i] = pointOri;
            coeffSelSurfVec[i] = coeff;
            laserCloudOriSurfFlag[i] = true;
          }
        }
      }
    }
  }

  void combineOptimizationCoeffs()
  {
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
    {
      if (laserCloudOriCornerFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]);
        coeffSel->push_back(coeffSelCornerVec[i]);
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
    {
      if (laserCloudOriSurfFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
  }

  bool LMOptimization(int iterCount)
  {
    // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    float srx = sin(transformTobeMapped[1]);
    float crx = cos(transformTobeMapped[1]);
    float sry = sin(transformTobeMapped[2]);
    float cry = cos(transformTobeMapped[2]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50)
    {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++)
    {
      // lidar -> camera
      pointOri.x = laserCloudOri->points[i].y;
      pointOri.y = laserCloudOri->points[i].z;
      pointOri.z = laserCloudOri->points[i].x;
      // lidar -> camera
      coeff.x = coeffSel->points[i].y;
      coeff.y = coeffSel->points[i].z;
      coeff.z = coeffSel->points[i].x;
      coeff.intensity = coeffSel->points[i].intensity;
      // in camera
      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

      float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

      float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeff.z;
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0)
    {

      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      isDegenerate = false;
      float eignThre[6] = { 100, 100, 100, 100, 100, 100 };
      for (int i = 5; i >= 0; i--)
      {
        if (matE.at<float>(0, i) < eignThre[i])
        {
          for (int j = 0; j < 6; j++)
          {
            matV2.at<float>(i, j) = 0;
          }
          isDegenerate = true;
        }
        else
        {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    if (isDegenerate)
    {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(
      pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
      pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
      pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(
      pow(matX.at<float>(3, 0) * 100, 2) +
      pow(matX.at<float>(4, 0) * 100, 2) +
      pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05)
    {
      return true; // converged
    }
    return false; // keep optimizing
  }

  void transformUpdate()
  {
    if (cloudInfo.imuAvailable == true)
    {
      if (std::abs(cloudInfo.imuPitchInit) < 1.4)
      {
        double imuWeight = imuRPYWeight;
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp roll
        transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
        imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
        // tf::Quaternion::slerp 实现球形线性插值，这里返回的结果是 transformQuaternion * (1 - imuWeight) + imuQuaternion * imuWeight
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[0] = rollMid;

        // slerp pitch
        transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
        imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[1] = pitchMid;
      }
    }

    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
  }

  float constraintTransformation(float value, float limit)
  {
    if (value < -limit)
      value = -limit;
    if (value > limit)
      value = limit;

    return value;
  }

  void saveKeyFramesAndFactor()
  {
    // 建图时还是保持原来的 saveFrame 判断标准，即只把当前帧的位姿与前一帧比，不考虑回到之前路过的地方的情况（因为建图大概只会走一遍）
    if (saveFrame() == false)
      return;

    // odom factor
    addOdomFactor();

    // loop factor
    addLoopFactor(); // radius search loop factor (I changed the orignal func name addLoopFactor to addLoopFactor)

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate); // 把 gtSAMgraph 里的因子加入到 isam 里，以 initialEstimate 为初始值对 isam 进行一次增量式优化
    isam->update(); // isam 因子图不变，只是基于上一次优化的结果，再对 isam 进行一次增量式优化。
    // gtsam::FixedLagSmoother 或 gtsam::BatchFixedLagSmoother 等类可以用来执行固定滞后平滑操作，即固定因子图里的一些因子，只对其他因子进行优化。
    // 具体操作方法为：将这些固定的变量的关键字和对应的值存储在一个 FixedLagSmootherParams 对象中，然后将该对象作为参数传递给 FixedLagSmoother 或 BatchFixedLagSmoother 对象的 smooth 函数
    // 注意这里用的优化器是 gtsam::ISAM2 而不是上面两个类

    // 如果检测到了回环，就对 isam 进行多次优化
    // Optional: 能不能写在回环检测线程里面，感觉写在这里会有点影响实时性？因为要多优化好几次
    if (aLoopIsClosed == true)
    {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisDenseKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
    pcl::copyPointCloud(*laserCloudDenseDS, *thisDenseKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    denseCloudKeyFrames.push_back(thisDenseKeyFrame);

    // make Scan Context for current key frame
    pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
    scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);

    // save sc data
    const auto& curr_scd = scManager.getConstRefRecentSCD();                       // @yan 返回最新的一个 sc
    std::string curr_scd_node_idx = padZeros(scManager.polarcontexts_.size() - 1); // @yan 输出 6 位数字的文件名
    saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);

    // save keyframe cloud
    pcl::io::savePCDFileBinaryCompressed(saveCornerScanDirectory + curr_scd_node_idx + ".pcd", *thisCornerKeyFrame);
    pcl::io::savePCDFileBinaryCompressed(saveSurfScanDirectory + curr_scd_node_idx + ".pcd", *thisSurfKeyFrame);

    pgTimeSaveStream << laserCloudRawTime << std::endl;

    // save path for visualization
    updatePath(thisPose6D);
  }

  bool saveFrame()
  {
    if (cloudKeyPoses3D->points.empty())
      return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
      transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
      abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
      abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
      sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
      return false;

    return true;
  }

  void addOdomFactor()
  {
    if (cloudKeyPoses3D->points.empty())
    {
      noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
      initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));

      writeVertex(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
      noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
      gtsam::Pose3 relPose = poseFrom.between(poseTo);
      gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), relPose, odometryNoise));
      initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);

      writeVertex(cloudKeyPoses3D->size(), poseTo);
      writeEdge({ cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size() }, relPose); // giseop
    }
  }

  void addLoopFactor()
  {
    if (loopIndexQueue.empty())
      return;

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
      int indexFrom = loopIndexQueue[i].first;
      int indexTo = loopIndexQueue[i].second;
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i]; // original
      auto noiseBetween = loopNoiseQueue[i]; // giseop for polymorhpism // shared_ptr<gtsam::noiseModel::Base>, typedef noiseModel::Base::shared_ptr gtsam::SharedNoiseModel
      gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));

      writeEdge({ indexFrom, indexTo }, poseBetween); // giseop
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();

    aLoopIsClosed = true;
  }

  void updatePath(const PointTypePose& pose_in)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
  }

  // 当检测到闭环时，重新更新 cloudKeyPoses3D 和 cloudKeyPoses6D，保证里面是经过图优化后的最新位姿
  void correctPoses()
  {
    if (cloudKeyPoses3D->points.empty())
      return;

    if (aLoopIsClosed == true)
    {
      // clear map cache
      // 一旦检测到了闭环，map 就要清空，因为 map 是基于里程计给的位姿制作的，现在闭环检测对里程计位姿做了修正，所以 map 也要重新制作
      // 相当于 laserCloudMapContainer 是一个基于里程计构建起来的 map，有闭环了它就要被修正
      laserCloudMapContainer.clear();
      // clear path
      globalPath.poses.clear();
      // update key poses
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i)
      {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        updatePath(cloudKeyPoses6D->points[i]);
      }

      aLoopIsClosed = false;
    }
  }

  void publishOdometry()
  {
    // Publish odometry for ROS (global)
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
    laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    poseStream << std::fixed << std::setprecision(4)
      << timeLaserInfoCur << " "
      << transformTobeMapped[3] << " " << transformTobeMapped[4] << " " << transformTobeMapped[5] << " "
      << transformTobeMapped[0] << " " << transformTobeMapped[1] << " " << transformTobeMapped[2]
      << std::endl;

    // Publish TF
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
    br.sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental)
    static bool lastIncreOdomPubFlag = false;       // 用于检查是否是第一次发布增量里程计
    static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3f increOdomAffine;         // incremental odometry in affine
    if (lastIncreOdomPubFlag == false)
    {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = trans2Affine3f(transformTobeMapped);
    }
    else
    {
      Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
      increOdomAffine = increOdomAffine * affineIncre; // 从这一句看出来，增量里程计输出的是从第一帧到当前帧的位姿变换，并非从上一帧到当前帧的位姿变换
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
      if (cloudInfo.imuAvailable == true)
      {
        if (std::abs(cloudInfo.imuPitchInit) < 1.4)
        {
          double imuWeight = 0.1;
          tf::Quaternion imuQuaternion;
          tf::Quaternion transformQuaternion;
          double rollMid, pitchMid, yawMid;

          // slerp roll
          transformQuaternion.setRPY(roll, 0, 0);
          imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          roll = rollMid;

          // slerp pitch
          transformQuaternion.setRPY(0, pitch, 0);
          imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          pitch = pitchMid;
        }
      }
      laserOdomIncremental.header.stamp = timeLaserInfoStamp;
      laserOdomIncremental.header.frame_id = odometryFrame;
      laserOdomIncremental.child_frame_id = "odom_mapping";
      laserOdomIncremental.pose.pose.position.x = x;
      laserOdomIncremental.pose.pose.position.y = y;
      laserOdomIncremental.pose.pose.position.z = z;
      laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
      if (isDegenerate)
        laserOdomIncremental.pose.covariance[0] = 1;
      else
        laserOdomIncremental.pose.covariance[0] = 0;
    }
    pubLaserOdometryIncremental.publish(laserOdomIncremental);
  }

  void publishFrames()
  {
    if (cloudKeyPoses3D->points.empty())
      return;
    // publish key poses
    publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
      *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
      publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
      publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish path
    if (pubPath.getNumSubscribers() != 0)
    {
      globalPath.header.stamp = timeLaserInfoStamp;
      globalPath.header.frame_id = odometryFrame;
      pubPath.publish(globalPath);
    }
  }

  /*************** lidar scan 回调函数结束 ***************/

  /*****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************
   *****************************************/

   /*************** 回环检测线程开始 ***************/

  void loopClosureThread()
  {
    if (loopClosureEnableFlag == false)
      return;

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
      rate.sleep();
      performLoopClousure();
      // performRSLoopClosure();
      // performSCLoopClosure(); // giseop
      visualizeLoopClosure();
    }
  }

  void performLoopClousure()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    copy_cloudKeyPoses2D->clear();            // giseop
    *copy_cloudKeyPoses2D = *cloudKeyPoses3D; // giseop
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    if (copy_cloudKeyPoses3D->size() == keyPoseNum)
      return;

    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    keyPoseNum = copy_cloudKeyPoses3D->size();
    pcl::PointCloud<PointType>::Ptr curKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr preKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyFrames(curKeyframeCloud, loopKeyCur, 0);
    if (curKeyframeCloud->size() < 300)
    {
      ROS_INFO_STREAM("Source cloud feature point num: " << curKeyframeCloud->size() << " < 300). Skip this keyframe.");
      return;
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(curKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());

    // pose transformation var
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame, tWrong, tCorrect;
    // pose graph var
    gtsam::Pose3 poseFrom, poseTo;


    /************** SC loop **************/
    bool scLoopFound = false;
    // find keys
    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int loopKeyPre = detectResult.first;
    float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)

    if (loopKeyPre != -1)
    {
      // get legitimate SC loop which satisfies the time and distance requirements
      if (pointDistance(copy_cloudKeyPoses3D->points[loopKeyCur], copy_cloudKeyPoses3D->points[loopKeyPre]) < historyKeyframeSearchRadius &&
        abs(copy_cloudKeyPoses6D->points[loopKeyPre].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
      {
        // extract cloud
        loopFindNearKeyFrames(preKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (preKeyframeCloud->size() < 1000)
        {
          ROS_INFO_STREAM("Target cloud feature point num: " << preKeyframeCloud->size() << " < 1000. Skip SC loop (" << loopKeyCur << ", " << loopKeyPre << ").");
        }
        else
        {
          if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, preKeyframeCloud, timeLaserInfoStamp, odometryFrame);

          // Align clouds
          icp.setInputTarget(preKeyframeCloud);
          icp.align(*unused_result);
          if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
          {
            ROS_INFO_STREAM("ICP fitness failed. SC loop (" << loopKeyCur << ", " << loopKeyPre << ") ICP score: "
              << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << ". Skip this SC loop.");
          }
          else
          {
            ROS_INFO_STREAM("ICP fitness passed. SC loop (" << loopKeyCur << ", " << loopKeyPre << ") ICP score: "
              << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << ". Add this SC loop.");
            scLoopFound = true;

            // publish corrected cloud
            if (pubIcpKeyFrames.getNumSubscribers() != 0)
            {
              pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
              pcl::transformPointCloud(*curKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
              publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
            }

            // Get pose transformation
            correctionLidarFrame = icp.getFinalTransformation();

            // transform from world origin to wrong pose
            tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
            // transform from world origin to corrected pose
            tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
            poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
            poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

            gtsam::Vector Vector6(6);
            float noiseScore = icp.getFitnessScore();
            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

            // Add pose constraint
            mtx.lock();
            loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();

            // add loop constriant
            loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
          }
        }
      }
    }

    /************** RS loop **************/
    // if no SC loop found, try to check if there is any RS loop
    if (scLoopFound == false)
    {
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) // 通过距离和时间判断是否可能存在闭环
        return;

      // extract cloud
      pcl::PointCloud<PointType>::Ptr preKeyframeCloud(new pcl::PointCloud<PointType>());
      loopFindNearKeyFrames(preKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
      if (preKeyframeCloud->size() < 1000)
      {
        ROS_INFO_STREAM("Target cloud feature point num: " << preKeyframeCloud->size() << " < 1000. Skip RS loop (" << loopKeyCur << ", " << loopKeyPre << ").");
        return;
      }
      if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        publishCloud(&pubHistoryKeyFrames, preKeyframeCloud, timeLaserInfoStamp, odometryFrame);

      // Align clouds
      icp.setInputTarget(preKeyframeCloud);
      icp.align(*unused_result);
      if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
      {
        ROS_INFO_STREAM("ICP fitness failed. RS loop (" << loopKeyCur << ", " << loopKeyPre << ") ICP score: "
          << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << ". Skip this RS loop.");
        return;
      }
      else
      {
        ROS_INFO_STREAM("ICP fitness passed. RS loop (" << loopKeyCur << ", " << loopKeyPre << ") ICP score: "
          << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << ". Add this RS loop.");
      }

      // publish corrected cloud
      if (pubIcpKeyFrames.getNumSubscribers() != 0)
      {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*curKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
      }

      // Get pose transformation
      correctionLidarFrame = icp.getFinalTransformation();
      // transform from world origin to wrong pose
      tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
      // transform from world origin to corrected pose
      tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
      pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
      poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
      poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

      gtsam::Vector Vector6(6);
      float noiseScore = icp.getFitnessScore();
      Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
      noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

      // Add pose constraint
      mtx.lock();
      loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
      loopPoseQueue.push_back(poseFrom.between(poseTo));
      loopNoiseQueue.push_back(constraintNoise);
      mtx.unlock();

      // add loop constriant
      loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
    }
  }

  void performRSLoopClosure()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    copy_cloudKeyPoses2D->clear();            // giseop
    *copy_cloudKeyPoses2D = *cloudKeyPoses3D; // giseop
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false) // 空接口，不用看
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) // 通过距离和时间判断是否可能存在闭环
        return;

    std::cout << "RS loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop

    // extract cloud
    pcl::PointCloud<PointType>::Ptr curKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr preKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyFrames(curKeyframeCloud, loopKeyCur, 0);
    loopFindNearKeyFrames(preKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
    if (curKeyframeCloud->size() < 300 || preKeyframeCloud->size() < 1000)
      return;
    if (pubHistoryKeyFrames.getNumSubscribers() != 0)
      publishCloud(&pubHistoryKeyFrames, preKeyframeCloud, timeLaserInfoStamp, odometryFrame);

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(curKeyframeCloud);
    icp.setInputTarget(preKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
      std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this RS loop." << std::endl;
      return;
    }
    else
    {
      std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this RS loop." << std::endl;
    }

    // publish corrected cloud
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*curKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    // loopIndexContainer[loopKeyCur] = loopKeyPre;
    loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
  }                                                                         // performRSLoopClosure

  void performSCLoopClosure()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;

    if (copy_cloudKeyPoses3D->size() == keyPoseNum)
      return;

    // find keys
    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    keyPoseNum = copy_cloudKeyPoses3D->size();

    int loopKeyPre = detectResult.first;
    float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
    if (loopKeyPre == -1 /* No loop found */)
      return;

    std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop

    // extract cloud
    pcl::PointCloud<PointType>::Ptr curKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr preKeyframeCloud(new pcl::PointCloud<PointType>());
    // @yan: try to make the icp process mathematically correct
    loopFindNearKeyFrames(curKeyframeCloud, loopKeyCur, 0);
    loopFindNearKeyFrames(preKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

    if (curKeyframeCloud->size() < 300 || preKeyframeCloud->size() < 1000)
    {
      cout << "[ICP points shortage] " << "Scan " << loopKeyCur << " can't perform icp because of points shortage." << endl;
      return;
    }
    if (pubHistoryKeyFrames.getNumSubscribers() != 0)
      publishCloud(&pubHistoryKeyFrames, preKeyframeCloud, timeLaserInfoStamp, odometryFrame);

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(curKeyframeCloud);
    icp.setInputTarget(preKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
    // giseop
    // TODO icp align with initial

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
      std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
      return;
    }
    else
    {
      std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
    }

    // publish corrected cloud
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*curKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // @yan: try to make the icp process mathematically correct
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

    // giseop, robust kernel for a SC loop
    float robustNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6);
    robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
    noiseModel::Base::shared_ptr robustConstraintNoise;
    robustConstraintNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),            // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(robustConstraintNoise);
    mtx.unlock();

    // add loop constriant
    // loopIndexContainer[loopKeyCur] = loopKeyPre;
    loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
  }                                                                         // performSCLoopClosure

  // 根据距离和时间初步判断是否可能存在闭环
  bool detectLoopClosureDistance(int* latestID, int* closestID)
  {
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
      return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop; // unused
    // kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    // kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)copy_cloudKeyPoses2D->size(); i++) // giseop
      copy_cloudKeyPoses2D->points[i].z = 1.1;                  // to relieve the z-axis drift, 1.1 is just foo val

    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D);                                                                                  // giseop
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0); // giseop

    // std::cout << "the number of RS-loop candidates  " << pointSearchIndLoop.size() << "." << std::endl; // giseop
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
      {
        loopKeyPre = id;
        break;
      }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  void loopFindNearKeyFrames(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize)
        continue;
      *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
      *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
      return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  void visualizeLoopClosure()
  {
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.scale.y = 0.1;
    markerEdge.scale.z = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
  }

  /*************** 下面是闭环检测线程中留给外部闭环检测器的接口，暂未投入使用 ***************/

  // 接受外部闭环检测器的回环消息，配合 detectLoopClosureExternal 使用，暂未使用
  void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
  {
    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopMsg->data.size() != 2)
      return;

    loopInfoVec.push_back(*loopMsg);

    while (loopInfoVec.size() > 5)
      loopInfoVec.pop_front();
  }

  // 通过外部闭环检测器检测闭环，仅做了个接口，暂未使用
  bool detectLoopClosureExternal(int* latestID, int* closestID)
  {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopInfoVec.empty())
      return false;

    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
      return false;

    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2)
      return false;

    // latest key
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i)
    {
      if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
        loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i)
    {
      if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
        loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    if (loopKeyCur == loopKeyPre)
      return false;

    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  /*************** 回环检测线程结束 ***************/

  /****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************/

  /*************** 地图可视化线程开始 ***************/

  void visualizeGlobalMapThread()
  {
    //
    ros::Rate rate(0.2);
    while (ros::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }

    if (saveMap == false)
      return;

    // save pose graph (runs when programe is closing)
    cout << "****************************************************" << endl;
    cout << "Saving the posegraph ..." << endl; // giseop

    for (auto& _line : vertices_str)
      pgSaveStream << _line << std::endl;
    for (auto& _line : edges_str)
      pgSaveStream << _line << std::endl;

    pgSaveStream.close();

    const std::string kitti_format_pg_filename{ mapDirectory + "optimized_poses.txt" };
    saveOptimizedVerticesKITTIformat(isamCurrentEstimate, kitti_format_pg_filename);

    // save map
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // save key frame transformations
    pcl::io::savePCDFileASCII(mapDirectory + "cloudKeyPoses3D.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileASCII(mapDirectory + "cloudKeyPoses6D.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalDenseMapCloud(new pcl::PointCloud<PointType>());
    keyPoseStream.open(mapDirectory + "keyFramePoses.txt");
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      *globalDenseMapCloud += *transformPointCloud(denseCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      // cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      // @yan: save key poses as txt file.
      // @yan: only here is the latest version of cloudKeyPoses6D. Save key poses at saveKeyFramesAndFactor() is not the ultimate version.
      keyPoseStream << std::fixed << std::setprecision(4)
        << cloudKeyPoses6D->points[i].time << " "
        << cloudKeyPoses6D->points[i].x << " " << cloudKeyPoses6D->points[i].y << " " << cloudKeyPoses6D->points[i].z << " "
        << cloudKeyPoses6D->points[i].roll << " " << cloudKeyPoses6D->points[i].pitch << " " << cloudKeyPoses6D->points[i].yaw
        << std::endl;
    }
    keyPoseStream.close();
    // down-sample and save corner cloud
    downSizeFilterCorner.setInputCloud(globalCornerCloud);
    downSizeFilterCorner.filter(*globalCornerCloudDS);
    pcl::io::savePCDFileBinaryCompressed(mapDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
    // down-sample and save surf cloud
    downSizeFilterSurf.setInputCloud(globalSurfCloud);
    downSizeFilterSurf.filter(*globalSurfCloudDS);
    pcl::io::savePCDFileBinaryCompressed(mapDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
    // save global point cloud map
    // here we use the source corner cloud and surf cloud (undownsampled)
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;
    pcl::io::savePCDFileBinaryCompressed(mapDirectory + "cloudGlobal.pcd", *globalMapCloud);
    // save dense point cloud map
    if (saveDenseMap)
      pcl::io::savePCDFileBinaryCompressed(mapDirectory + "cloudDense.pcd", *globalDenseMapCloud);
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed" << endl;
  }

  void publishGlobalMap()
  {
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
      return;

    if (cloudKeyPoses3D->points.empty() == true)
      return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());

    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
      if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
        continue;
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
      *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
  }

  /*************** 地图可视化线程结束 ***************/

  /****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************
  *****************************************/

  /*************** 下面是一些 util 函数 ***************/

  void pointAssociateToMap(PointType const* const pi, PointType* const po)
  {
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType* pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
      pointFrom = &cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
  {
    /* RzRyRx 用的是 intrinsic rotation 的定义，且坐标系的旋转顺序为 zyx，但该函数传入的参数顺序为 roll（绕 x 轴的转角）, pitch（绕 y 轴的转角）, yaw（绕 z 轴的转角）
    在 gtsam 源代码 (https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Rot3Q.cpp) 里看到 RzRyRx 函数的定义:

    Rot3 Rot3::RzRyRx(double x, double y, double z, OptionalJacobian<3, 1> Hx,
                      OptionalJacobian<3, 1> Hy, OptionalJacobian<3, 1> Hz) {
      ...
      return Rot3(
          gtsam::Quaternion(Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ())) *
          gtsam::Quaternion(Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY())) *
          gtsam::Quaternion(Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX())));
    }

    从上述代码可以看出，RzRyRx 函数表示的是 zyx 顺序的 intrinsic rotation 或 xyz 顺序的 extrinsic rotation
    */
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
      gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
  }

  gtsam::Pose3 trans2gtsamPose(float transformIn[])
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
      gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }

  Eigen::Affine3f trans2Affine3f(float transformIn[])
  {
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
  }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  void writeVertex(const int _node_idx, const gtsam::Pose3& _initPose)
  {
    gtsam::Point3 t = _initPose.translation();
    gtsam::Rot3 R = _initPose.rotation();

    std::string curVertexInfo{
      "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " " + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " " + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    vertices_str.emplace_back(curVertexInfo);
  }

  void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose)
  {
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo{
      "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " " + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z()) + " " + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    edges_str.emplace_back(curEdgeInfo);
  }
  /*************** util 函数结束 ***************/
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sc_lio_sam");

  mapOptimization MO;

  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

  std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

  ros::spin();

  loopthread.join();
  visualizeMapThread.join();

  return 0;
}
