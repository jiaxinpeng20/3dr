
size_t FindBestMatchesOneWayBruteForce(const Eigen::MatrixXi& dists,
                                       const float max_ratio,
                                       const float max_distance,
                                       std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (Eigen::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const int dist = dists(i1, i2);
      if (dist > best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

void FindBestMatchesBruteForce(const Eigen::MatrixXi& dists,
                               const float max_ratio, const float max_distance,
                               const bool cross_check,
                               FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayBruteForce(
      dists, max_ratio, max_distance, &matches12);

  if (cross_check) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayBruteForce(
        dists.transpose(), max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

// Mutexes that ensure that only one thread extracts/matches on the same GPU
// at the same time, since SiftGPU internally uses static variables.
static std::map<int, std::unique_ptr<std::mutex>> sift_extraction_mutexes;
static std::map<int, std::unique_ptr<std::mutex>> sift_matching_mutexes;

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter) {
  if (guided_filter != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(keypoints1->size(), descriptors1.rows());
    CHECK_EQ(keypoints2->size(), descriptors2.rows());
  }

  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
      descriptors1.rows(), descriptors2.rows());

  for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
      if (guided_filter != nullptr &&
          guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                        (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
        dists(i1, i2) = 0;
      } else {
        dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
      }
    }
  }

  return dists;
}

void FindNearestNeighborsFLANN(
    const FeatureDescriptors& query, const FeatureDescriptors& database,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        indices,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        distances) {
  if (query.rows() == 0 || database.rows() == 0) {
    return;
  }

  const size_t kNumNearestNeighbors = 2;
  const size_t kNumTreesInForest = 4;

  const size_t num_nearest_neighbors =
      std::min(kNumNearestNeighbors, static_cast<size_t>(database.rows()));

  indices->resize(query.rows(), num_nearest_neighbors);
  distances->resize(query.rows(), num_nearest_neighbors);
  const flann::Matrix<uint8_t> query_matrix(const_cast<uint8_t*>(query.data()),
                                            query.rows(), 128);
  const flann::Matrix<uint8_t> database_matrix(
      const_cast<uint8_t*>(database.data()), database.rows(), 128);

  flann::Matrix<int> indices_matrix(indices->data(), query.rows(),
                                    num_nearest_neighbors);
  std::vector<float> distances_vector(query.rows() * num_nearest_neighbors);
  flann::Matrix<float> distances_matrix(distances_vector.data(), query.rows(),
                                        num_nearest_neighbors);
  flann::Index<flann::L2<uint8_t>> index(
      database_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
  index.buildIndex();
  index.knnSearch(query_matrix, indices_matrix, distances_matrix,
                  num_nearest_neighbors, flann::SearchParams(128));

  for (Eigen::Index query_index = 0; query_index < indices->rows();
       ++query_index) {
    for (Eigen::Index k = 0; k < indices->cols(); ++k) {
      const Eigen::Index database_index = indices->coeff(query_index, k);
      distances->coeffRef(query_index, k) =
          query.row(query_index)
              .cast<int>()
              .dot(database.row(database_index).cast<int>());
    }
  }
}

size_t FindBestMatchesOneWayFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances,
    const float max_ratio, const float max_distance,
    std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(indices.rows(), -1);

  for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (int n_idx = 0; n_idx < indices.cols(); ++n_idx) {
      const int d2_idx = indices(d1_idx, n_idx);
      const int dist = distances(d1_idx, n_idx);
      if (dist > best_dist) {
        best_i2 = d2_idx;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[d1_idx] = best_i2;
  }

  return num_matches;
}

void FindBestMatchesFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_2to1,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_2to1,
    const float max_ratio, const float max_distance, const bool cross_check,
    FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayFLANN(
      indices_1to2, distances_1to2, max_ratio, max_distance, &matches12);

  if (cross_check && indices_2to1.rows()) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayFLANN(
        indices_2to1, distances_2to1, max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void CFeatureExtraction::FeatureImport(onst ImageReaderOptions& reader_options,
                                 const std::string& import_path)
{
      PrintHeading1("Feature import");

  if (!ExistsDir(import_path_)) {
    std::cerr << "  ERROR: Import directory does not exist." << std::endl;
    return;
  }

  Database database(reader_options_.database_path);
  ImageReader image_reader(reader_options_, &database);

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Processing file [%d/%d]",
                              image_reader.NextIndex() + 1,
                              image_reader.NumImages())
              << std::endl;

    // Load image data and possibly save camera to database.
    Camera camera;
    Image image;
    Bitmap bitmap;
    if (image_reader.Next(&camera, &image, &bitmap, nullptr) !=
        ImageReader::Status::SUCCESS) {
      continue;
    }

    const std::string path = JoinPaths(import_path_, image.Name() + ".txt");

    if (ExistsFile(path)) {
      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

      std::cout << "  Features:       " << keypoints.size() << std::endl;

      DatabaseTransaction database_transaction(&database);

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database.WriteImage(image));
      }

      if (!database.ExistsKeypoints(image.ImageId())) {
        database.WriteKeypoints(image.ImageId(), keypoints);
      }

      if (!database.ExistsDescriptors(image.ImageId())) {
        database.WriteDescriptors(image.ImageId(), descriptors);
      }
    } else {
      std::cout << "  SKIP: No features found at " << path << std::endl;
    }
  }

  GetTimer().PrintMinutes();
    }
}

void CFeatureExtraction::WarnDarknessAdaptivityNotAvailable() {
  std::cout << "WARNING: Darkness adaptivity only available for GLSL SiftGPU."
            << std::endl;
}

}  // namespace

bool CFeatureExtraction::SiftExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  if (domain_size_pooling) {
    CHECK_OPTION_GT(dsp_min_scale, 0);
    CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
    CHECK_OPTION_GT(dsp_num_scales, 0);
  }
  return true;
}

bool SiftMatchingOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  CHECK_OPTION_GT(max_error, 0.0);
  CHECK_OPTION_GE(min_num_trials, 0);
  CHECK_OPTION_GT(max_num_trials, 0);
  CHECK_OPTION_LE(min_num_trials, max_num_trials);
  CHECK_OPTION_GE(min_inlier_ratio, 0);
  CHECK_OPTION_LE(min_inlier_ratio, 1);
  CHECK_OPTION_GE(min_num_inliers, 0);
  return true;
}

bool CFeatureExtraction::ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  CHECK(!options.estimate_affine_shape);
  CHECK(!options.domain_size_pooling);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(bitmap.Width(), bitmap.Height(), options.num_octaves,
                  options.octave_resolution, options.first_octave),
      &vl_sift_delete);
  if (!sift) {
    return false;
  }

  vl_sift_set_peak_thresh(sift.get(), options.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), options.edge_threshold);

  // Iterate through octaves.
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptors> level_descriptors;
  bool first_octave = true;
  while (true) {
    if (first_octave) {
      const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      if (vl_sift_process_first_octave(sift.get(), data_float.data())) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift.get())) {
        break;
      }
    }

    // Detect keypoints.
    vl_sift_detect(sift.get());

    // Extract detected keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift.get());
    const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level.
    size_t level_idx = 0;
    int prev_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level.
          level_keypoints.back().resize(level_idx);
          if (descriptors != nullptr) {
            level_descriptors.back().conservativeResize(level_idx, 128);
          }
        }

        // Add containers for new DOG level.
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(options.max_num_orientations *
                                     num_keypoints);
        if (descriptors != nullptr) {
          level_descriptors.emplace_back(
              options.max_num_orientations * num_keypoints, 128);
        }
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (options.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(
            sift.get(), angles, &vl_keypoints[i]);
      }

      // Note that this is different from SiftGPU, which selects the top
      // global maxima as orientations while this selects the first two
      // local maxima. It is not clear which procedure is better.
      const int num_used_orientations =
          std::min(num_orientations, options.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx] =
            FeatureKeypoint(vl_keypoints[i].x + 0.5f, vl_keypoints[i].y + 0.5f,
                            vl_keypoints[i].sigma, angles[o]);
        if (descriptors != nullptr) {
          Eigen::MatrixXf desc(1, 128);
          vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                           &vl_keypoints[i], angles[o]);
          if (options.normalization ==
              SiftExtractionOptions::Normalization::L2) {
            desc = L2NormalizeFeatureDescriptors(desc);
          } else if (options.normalization ==
                     SiftExtractionOptions::Normalization::L1_ROOT) {
            desc = L1RootNormalizeFeatureDescriptors(desc);
          } else {
            LOG(FATAL) << "Normalization type not supported";
          }

          level_descriptors.back().row(level_idx) =
              FeatureDescriptorsToUnsignedByte(desc);
        }

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave.
    level_keypoints.back().resize(level_idx);
    if (descriptors != nullptr) {
      level_descriptors.back().conservativeResize(level_idx, 128);
    }
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option.
  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = level_keypoints.size() - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += level_keypoints[i].size();
    if (num_features > options.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.
  {
    size_t k = 0;
    keypoints->resize(num_features_with_orientations);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        (*keypoints)[k] = level_keypoints[i][j];
        k += 1;
      }
    }
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    size_t k = 0;
    descriptors->resize(num_features_with_orientations, 128);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        descriptors->row(k) = level_descriptors[i].row(j);
        k += 1;
      }
    }
    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

bool CFeatureExtraction::ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                                     const Bitmap& bitmap,
                                     FeatureKeypoints* keypoints,
                                     FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup covariant SIFT detector.
  std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
  if (!covdet) {
    return false;
  }

  const int kMaxOctaveResolution = 1000;
  CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

  vl_covdet_set_first_octave(covdet.get(), options.first_octave);
  vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
  vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
  vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

  {
    const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
    std::vector<float> data_float(data_uint8.size());
    for (size_t i = 0; i < data_uint8.size(); ++i) {
      data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
    vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
                        bitmap.Height());
  }

  vl_covdet_detect(covdet.get(), options.max_num_features);

  if (!options.upright) {
    if (options.estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }
  }

  const int num_features = vl_covdet_get_num_features(covdet.get());
  VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

  // Sort features according to detected octave and scale.
  std::sort(
      features, features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        } else {
          return feature1.o > feature2.o;
        }
      });

  const size_t max_num_features = static_cast<size_t>(options.max_num_features);

  // Copy detected keypoints and clamp when maximum number of features reached.
  int prev_octave_scale_idx = std::numeric_limits<int>::max();
  for (int i = 0; i < num_features; ++i) {
    FeatureKeypoint keypoint;
    keypoint.x = features[i].frame.x + 0.5;
    keypoint.y = features[i].frame.y + 0.5;
    keypoint.a11 = features[i].frame.a11;
    keypoint.a12 = features[i].frame.a12;
    keypoint.a21 = features[i].frame.a21;
    keypoint.a22 = features[i].frame.a22;
    keypoints->push_back(keypoint);

    const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
    CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

    if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints->size() >= max_num_features) {
      break;
    }

    prev_octave_scale_idx = octave_scale_idx;
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    descriptors->resize(keypoints->size(), 128);

    const size_t kPatchResolution = 15;
    const size_t kPatchSide = 2 * kPatchResolution + 1;
    const double kPatchRelativeExtent = 7.5;
    const double kPatchRelativeSmoothing = 1;
    const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
    const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

    std::vector<float> patch(kPatchSide * kPatchSide);
    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

    float dsp_min_scale = 1;
    float dsp_scale_step = 0;
    int dsp_num_scales = 1;
    if (options.domain_size_pooling) {
      dsp_min_scale = options.dsp_min_scale;
      dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
                       options.dsp_num_scales;
      dsp_num_scales = options.dsp_num_scales;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
        scaled_descriptors(dsp_num_scales, 128);

    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    if (!sift) {
      return false;
    }

    vl_sift_set_magnif(sift.get(), 3.0);

    for (size_t i = 0; i < keypoints->size(); ++i) {
      for (int s = 0; s < dsp_num_scales; ++s) {
        const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

        VlFrameOrientedEllipse scaled_frame = features[i].frame;
        scaled_frame.a11 *= dsp_scale;
        scaled_frame.a12 *= dsp_scale;
        scaled_frame.a21 *= dsp_scale;
        scaled_frame.a22 *= dsp_scale;

        vl_covdet_extract_patch_for_frame(
            covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
            kPatchRelativeSmoothing, scaled_frame);

        vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
                              2 * kPatchSide, patch.data(), kPatchSide,
                              kPatchSide, kPatchSide);

        vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
                                    scaled_descriptors.row(s).data(),
                                    kPatchSide, kPatchSide, kPatchResolution,
                                    kPatchResolution, kSigma, 0);
      }

      Eigen::Matrix<float, 1, 128> descriptor;
      if (options.domain_size_pooling) {
        descriptor = scaled_descriptors.colwise().mean();
      } else {
        descriptor = scaled_descriptors;
      }

      if (options.normalization == SiftExtractionOptions::Normalization::L2) {
        descriptor = L2NormalizeFeatureDescriptors(descriptor);
      } else if (options.normalization ==
                 SiftExtractionOptions::Normalization::L1_ROOT) {
        descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
      } else {
        LOG(FATAL) << "Normalization type not supported";
      }

      descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
    }

    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}
