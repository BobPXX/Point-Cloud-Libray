//3D Object Recognition based on Correspondence Grouping: https://pcl.readthedocs.io/projects/tutorials/en/latest/correspondence_grouping.html#correspondence-grouping

#include <iostream>
#include <string>
#include <set>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/board.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <ctime>
#include <string>
clock_t start, end;
typedef pcl::PointXYZ PointT;

void convert_XYZRGBNormal_to_Normal(const pcl::PointCloud<pcl::PointXYZRGBNormal>& in, pcl::PointCloud<pcl::Normal>& out)
{
    int M = in.points.size();
    for (int i = 0; i < M; i++)
    {
        pcl::Normal p;
        p.normal_x = in.points[i].normal_x;
        p.normal_y = in.points[i].normal_y;
        p.normal_z = in.points[i].normal_z;
        p.curvature = in.points[i].curvature;
        out.points.push_back(p);
    }
    out.width = 1;
    out.height = M;
}

void convert(const pcl::PointCloud<pcl::PointXYZRGBNormal>& in, pcl::PointCloud<pcl::PointXYZRGBA>& out) {
    out.width = in.width;
    out.height = in.height;

    out.reserve(in.size());

    std::transform(
        std::begin(in),
        std::end(in),
        std::back_inserter(out),
        [](const pcl::PointXYZRGBNormal& point) {
            return pcl::PointXYZRGBA{ point.x, point.y, point.z, point.r, point.g, point.b, point.a };
        }
    );
}

void convert_no_color(const pcl::PointCloud<pcl::PointXYZRGBA>& in, pcl::PointCloud<pcl::PointXYZ>& out)
{
    int M = in.points.size();
    for (int i = 0; i < M; i++)
    {
        pcl::PointXYZ p;
        p.x = in.points[i].x;
        p.y = in.points[i].y;
        p.z = in.points[i].z;
        out.points.push_back(p);
    }
    out.width = 1;
    out.height = M;
}

void down_sampling(pcl::PointCloud<PointT>::Ptr in, float down_sampling_leafsize)
{
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(in);
    sor.setLeafSize(down_sampling_leafsize, down_sampling_leafsize, down_sampling_leafsize);
    sor.filter(*in);
}

void down_sampling_ori(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in, float down_sampling_leafsize)
{
    pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
    sor.setInputCloud(in);
    sor.setLeafSize(down_sampling_leafsize, down_sampling_leafsize, down_sampling_leafsize);
    sor.filter(*in);
}

bool is_same_obj(pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<PointT>::Ptr obj)
{
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(scene);
    icp.setInputTarget(obj);

    pcl::PointCloud<PointT> Final;
    icp.align(Final);
    return icp.hasConverged();
}

void GeometricConsistency(pcl::PointCloud<PointT>::Ptr& obj_keypoints, pcl::PointCloud<PointT>::Ptr& data_keypoints,
    pcl::CorrespondencesPtr& model_scene_corrs, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& rototranslations,
    std::vector<pcl::Correspondences>& clustered_corrs, float GC_size, float GC_threshold)
{
    pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
    gc_clusterer.setGCSize(GC_size);        //consensus set resolution, default = resolution
    gc_clusterer.setGCThreshold(GC_threshold); //minimum cluster size

    gc_clusterer.setInputCloud(obj_keypoints);
    gc_clusterer.setSceneCloud(data_keypoints);
    gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize(rototranslations, clustered_corrs);
}

void Hough3D(pcl::PointCloud<PointT>::Ptr& data, pcl::PointCloud<PointT>::Ptr& obj,
    pcl::PointCloud<pcl::Normal>::Ptr& data_normal, pcl::PointCloud<pcl::Normal>::Ptr& obj_normal,
    pcl::PointCloud<PointT>::Ptr& data_keypoints, pcl::PointCloud<PointT>::Ptr& obj_keypoints,
    pcl::CorrespondencesPtr& model_scene_corrs, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& rototranslations,
    std::vector<pcl::Correspondences>& clustered_corrs, float Hough_R, float bin_size, float Hough_threshold)
{
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf(new pcl::PointCloud<pcl::ReferenceFrame>());

    pcl::BOARDLocalReferenceFrameEstimation<PointT, pcl::Normal, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(Hough_R);//估计局部参考坐标系时当前点的邻域搜索半径 default = resolution

    rf_est.setInputCloud(obj_keypoints);
    rf_est.setInputNormals(obj_normal);
    rf_est.setSearchSurface(obj);
    rf_est.compute(*model_rf);

    rf_est.setInputCloud(data_keypoints);
    rf_est.setInputNormals(data_normal);
    rf_est.setSearchSurface(data);
    rf_est.compute(*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
    clusterer.setHoughBinSize(bin_size);     //hough空间的采样间隔：0.01
    clusterer.setHoughThreshold(Hough_threshold); //在hough空间确定是否有实例存在的最少票数阈值

    clusterer.setUseInterpolation(true);     //设置是否对投票在hough空间进行插值计算
    clusterer.setUseDistanceWeight(false);   //设置在投票时是否将对应点之间的距离作为权重参与计算

    clusterer.setInputCloud(obj_keypoints); //设置模型点云的关键点
    clusterer.setInputRf(model_rf);           //设置模型对应的局部坐标系
    clusterer.setSceneCloud(data_keypoints);
    clusterer.setSceneRf(scene_rf);
    clusterer.setModelSceneCorrespondences(model_scene_corrs);//设置模型与场景的对应点的集合

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize(rototranslations, clustered_corrs); //结果包含变换矩阵和对应点聚类结果
}

void del_plane(pcl::PointCloud<PointT>::Ptr data, float distance)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);	//定义模型系数
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);					//定义点索引
    pcl::SACSegmentation<PointT> seg;								//创建分割对象
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);									//设置分割对象所选用的模型类型
    seg.setMethodType(pcl::SAC_RANSAC);									//设置算法类型
    seg.setMaxIterations(100000);
    seg.setDistanceThreshold(distance);	//unit:m									//本算法中唯一一个参数，设置距离阈值
    seg.setInputCloud(data);												//设置输入点云
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(data);
    extract.setIndices(inliers);
    extract.setNegative(true);	/// 设置为true，删除索引中的点
    extract.filter(*data);
}

void del_plane_ori(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr data, float distance)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);	//定义模型系数
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);					//定义点索引
    pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;								//创建分割对象
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);									//设置分割对象所选用的模型类型
    seg.setMethodType(pcl::SAC_RANSAC);									//设置算法类型
    seg.setMaxIterations(100000);
    seg.setDistanceThreshold(distance);	//unit:m									//本算法中唯一一个参数，设置距离阈值
    seg.setInputCloud(data);												//设置输入点云
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(data);
    extract.setIndices(inliers);
    extract.setNegative(true);	/// 设置为true，删除索引中的点
    extract.filter(*data);
}


void denoising(pcl::PointCloud<PointT>::Ptr data)
{
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(data);
    sor.setMeanK(100);
    sor.setStddevMulThresh(1);
    sor.filter(*data);
}

void denoising_ori(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr data)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
    sor.setInputCloud(data);
    sor.setMeanK(100);
    sor.setStddevMulThresh(1);
    sor.filter(*data);
}

void show_figure(pcl::PointCloud<PointT>::Ptr pc)
{
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addPointCloud<pcl::PointXYZ>(pc, "point cloud");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}

void delete_paired_points(pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<PointT>::Ptr paired_points, float radis)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ExtractIndices<PointT> extract;
    std::set<std::size_t> index;
    cout << "deleteing..." << endl;
    for (std::size_t i = 0; i < paired_points->size(); ++i)
    {
        //if (i % 100 == 0)
        //    cout << "Total: " << paired_points->size() << ", now: " << i << endl;
        for (std::size_t j = 0; j < scene->size(); ++j)
        {
            if (index.count(j))
                continue;
            if (pow((paired_points->points[i].x - scene->points[j].x), 2) +
                pow((paired_points->points[i].y - scene->points[j].y), 2) +
                pow((paired_points->points[i].z - scene->points[j].z), 2) < radis)
            {
                index.insert(j);
            }
        }
    }
    for (auto i : index)
        inliers->indices.push_back(i);

    extract.setInputCloud(scene);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*scene);
    denoising(scene);
    pcl::io::savePLYFileASCII("atfer_delete.ply", *scene);
    cout << "delete finish!" << endl;
    cout << "there are " << scene->width * scene->height << " data points left" << endl;
}

double computeCloudResolution(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(cloud);

    for (std::size_t i = 0; i < cloud->size(); ++i)
    {
        if (!std::isfinite((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt(sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}


int main()
{
    float down_sampling_leafsize_scene = 0.4f;
    float down_sampling_leafsize_obj = 0.55f;

    float del_plane_distance_scene = 12.5 * down_sampling_leafsize_scene;
    //float del_plane_distance_obj = 10 * down_sampling_leafsize_obj;

    //N nearest neighbors of each point in Ksearch for calculating normal
    int obj_nearest_points = 10;
    int scene_nearest_points = 10;

    //search radius for keypoints extraction
    float keypoint_R_scene = 5 * down_sampling_leafsize_scene; //default = resolution
    float keypoint_R_obj = 4 * down_sampling_leafsize_obj; //default = resolution

    //search radius for description
    float des_R_scene = 5 * keypoint_R_scene; //default = resolution
    float des_R_obj = 5 * keypoint_R_scene; //default = resolution

    //matching threshold in KdTree coresspondence search
    float matching_threshold = 0.5f;
    /*
    //GC custering algorithm
    float GC_size = 5.0f; //consensus set resolution, default = resolution
    float GC_threshold = 150.0f; //minimum cluster size
    */

    //Hough voting; 
    float Hough_R = 1.0 * des_R_scene; //for reference frame
    float bin_size = 1.0 * des_R_scene;
    float Hough_threshold = 25;


    start = clock();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_ori(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    const auto ret1 = pcl::io::loadPLYFile("./scene/scene_black_1/test1.ply", *scene_ori);  //single:./scene/scene_black_1/test1.ply; 4objects:./scene/scene_mix/all.ply
    if (ret1 < 0) {
        std::cerr << "Failed to reading Polygon mesh data" << std::endl;
        return false;
    }

    std::cout << "The ori scene has: " << scene_ori->width * scene_ori->height
        << " data points (" << pcl::getFieldsList(*scene_ori) << ")." << std::endl;

    down_sampling_ori(scene_ori, down_sampling_leafsize_scene);
    del_plane_ori(scene_ori, del_plane_distance_scene);
    denoising_ori(scene_ori);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_with_color(new pcl::PointCloud<pcl::PointXYZRGBA>);
    convert(*scene_ori, *scene_with_color);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
    convert_no_color(*scene_with_color, *scene);

    //calculate normal for scene
    /*
    pcl::PointCloud<pcl::Normal>::Ptr scene_normal(new pcl::PointCloud<pcl::Normal>);
    convert_XYZRGBNormal_to_Normal(*scene_ori, *scene_normal);
    pcl::io::savePLYFileASCII("scene_normal.ply", *scene_normal);
    */

    pcl::search::KdTree<PointT>::Ptr scene_kdtree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normal(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimationOMP<PointT, pcl::Normal> scene_norm_est;
    scene_norm_est.setSearchMethod(scene_kdtree);
    //norm_est.setNumberOfThreads(4);   //手动设置线程数
    scene_norm_est.setKSearch(scene_nearest_points);
    scene_norm_est.setInputCloud(scene);
    scene_norm_est.compute(*scene_normal);
    pcl::io::savePLYFileASCII("scene_normal.ply", *scene_normal);



    /*
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr obj_ori(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr obj_with_color(new pcl::PointCloud<pcl::PointXYZRGBA>);
    const auto ret2 = pcl::io::loadPLYFile("./scene/scene_black_1/test1.ply", *obj_ori);
    if (ret2 < 0) {
        std::cerr << "Failed to reading Polygon mesh data" << std::endl;
        return false;
    }
    */

    pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
    const auto ret2 = pcl::io::loadPLYFile("./model/FCD_black3.ply", *obj);
    if (ret2 < 0) {
        std::cerr << "Failed to reading Polygon mesh data" << std::endl;
        return false;
    }

    std::cout << "The ori model has: " << obj->width * obj->height
        << " data points (" << pcl::getFieldsList(*obj) << ")." << std::endl;

    down_sampling(obj, down_sampling_leafsize_obj);
    //del_plane(obj, del_plane_distance_obj);
    denoising(obj);
    pcl::io::savePLYFileASCII("sampled_model.ply", *obj);

    //calculate normal for model
    pcl::search::KdTree<PointT>::Ptr obj_kdtree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr obj_normal(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimationOMP<PointT, pcl::Normal> obj_norm_est;
    obj_norm_est.setSearchMethod(obj_kdtree);
    //norm_est.setNumberOfThreads(4);   //手动设置线程数
    obj_norm_est.setKSearch(obj_nearest_points);
    obj_norm_est.setInputCloud(obj);
    obj_norm_est.compute(*obj_normal);
    pcl::io::savePLYFileASCII("obj_normal.ply", *obj_normal);

    std::cout << "After preprocessing, the scene has: " << scene->width * scene->height
        << " data points (" << pcl::getFieldsList(*scene) << ")." << std::endl;
    pcl::io::savePLYFileASCII("processed_scene.ply", *scene);
    std::cout << "After preprocessing, the model has: " << obj->width * obj->height
        << " data points (" << pcl::getFieldsList(*obj) << ")." << std::endl;
    pcl::io::savePLYFileASCII("processed_obj.ply", *obj);

    end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Process Time:" << endtime << endl;

    /*
    pcl::visualization::CloudViewer viewer("PCL Viewer");
    viewer.showCloud(obj, "cloud");
    while (!viewer.wasStopped())
    {
    }
    */

    start = clock();

    // keypoints extraction
    pcl::UniformSampling<PointT> uniform_sampling;
    uniform_sampling.setInputCloud(scene);        //输入点云
    uniform_sampling.setRadiusSearch(keypoint_R_scene);  //输入半径
    pcl::PointCloud<PointT>::Ptr scene_keypoints(new pcl::PointCloud<PointT>());
    uniform_sampling.filter(*scene_keypoints);   //滤波
    std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

    uniform_sampling.setInputCloud(obj);
    uniform_sampling.setRadiusSearch(keypoint_R_obj);
    pcl::PointCloud<PointT>::Ptr obj_keypoints(new pcl::PointCloud<PointT>());
    uniform_sampling.filter(*obj_keypoints);
    std::cout << "model total points: " << obj->size() << "; Selected Keypoints: " << obj_keypoints->size() << std::endl;


    //use manually selected keypoints
    /*
    pcl::PointCloud<PointT>::Ptr scene_keypoints(new pcl::PointCloud<PointT>());
    const auto ret3 = pcl::io::loadPLYFile("./manual_select/scene1/Picking_list_3.ply", *scene_keypoints);
    if (ret3 < 0) {
        std::cerr << "Failed to reading scene keypoints" << std::endl;
        return false;
    }
    std::cout << "scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

    pcl::PointCloud<PointT>::Ptr obj_keypoints(new pcl::PointCloud<PointT>());
    const auto ret4 = pcl::io::loadPLYFile("./manual_select/model/Picking_list_3.ply", *obj_keypoints);
    if (ret4 < 0) {
        std::cerr << "Failed to reading model keypoints" << std::endl;
        return false;
    }
    std::cout << "model total points: " << obj->size() << "; Selected Keypoints: " << obj_keypoints->size() << std::endl;
    */


    // calculate descriptor
    pcl::SHOTEstimationOMP < PointT, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch(des_R_obj);     //设置搜索半径

    pcl::PointCloud<pcl::SHOT352>::Ptr obj_des(new pcl::PointCloud<pcl::SHOT352>());
    descr_est.setInputCloud(obj_keypoints);  //模型点云的关键点
    descr_est.setInputNormals(obj_normal);  //模型点云的法线 
    descr_est.setSearchSurface(obj);         //模型点云       
    descr_est.compute(*obj_des);     //计算描述子

    descr_est.setRadiusSearch(des_R_scene);
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_des(new pcl::PointCloud<pcl::SHOT352>());
    descr_est.setInputCloud(scene_keypoints);
    descr_est.setInputNormals(scene_normal);
    descr_est.setSearchSurface(scene);
    descr_est.compute(*scene_des);

    // Find Model-Scene Correspondences with KdTree
    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
    pcl::KdTreeFLANN<pcl::SHOT352> match_search; //设置配准方式
    match_search.setInputCloud(obj_des);//模型点云的描述子

    for (std::size_t i = 0; i < scene_des->size(); ++i)
    {
        std::vector<int> neigh_indices(1);    //设置最近邻点的索引
        std::vector<float> neigh_sqr_dists(1);//设置最近邻平方距离值
        if (!std::isfinite(scene_des->at(i).descriptor[0])) //忽略 NaNs点
            continue;
        int found_neighs = match_search.nearestKSearch(scene_des->at(i), 1, neigh_indices, neigh_sqr_dists);
        if (found_neighs == 1 && neigh_sqr_dists[0] < matching_threshold) //仅当描述子与临近点的平方距离小于matching_threshold（描述子与临近的距离在一般在0到1之间）才添加匹配
        {
            //neigh_indices[0]给定点，i是配准数neigh_sqr_dists[0]与临近点的平方距离
            pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back(corr);//把配准的点存储在容器中
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;


    // correspondence clustering
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //correspondence clustering algorithm
    //GeometricConsistency(obj_keypoints, scene_keypoints, model_scene_corrs, rototranslations, clustered_corrs, GC_size, GC_threshold);
    Hough3D(scene, obj, scene_normal, obj_normal, scene_keypoints, obj_keypoints, model_scene_corrs, rototranslations, clustered_corrs, Hough_R, bin_size, Hough_threshold);
    std::cout << "Model instances found: " << rototranslations.size() << std::endl;
    for (size_t i = 0; i < rototranslations.size(); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

        //打印处相对于输入模型的旋转矩阵与平移矩阵
        Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

        printf("\n");
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
        printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
        printf("\n");
        printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
    }
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Keypoints/descriptor/Correspondences/Clustering Time:" << endtime << endl;



    //Visualization
    pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
    viewer.addPointCloud(scene, "scene_cloud");

    pcl::PointCloud<PointT>::Ptr off_scene_model(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointT>());

    // show correspondences, keypoints
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud(*obj, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
    pcl::transformPointCloud(*obj_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
    viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");


    pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_keypoints_color_handler(scene_keypoints, 0, 0, 255);
    viewer.addPointCloud(scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");


    for (std::size_t i = 0; i < rototranslations.size(); ++i)
    {
        pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*obj, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointT> rotated_model_color_handler(rotated_model, 255, 0, 0); //255,0,0
        viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());


        for (std::size_t j = 0; j < clustered_corrs[i].size(); ++j)
        {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << i << "_" << j;
            PointT& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
            PointT& scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer.addLine<PointT, PointT>(model_point, scene_point, 0, 255, 0, ss_line.str());
        }

    }
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
    return (0);




}


