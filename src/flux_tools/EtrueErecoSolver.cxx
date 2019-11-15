#include "BargerPropagator.h"

#include "Eigen/Dense"
#include "Eigen/StdVector"
#include "Eigen/Sparse"
// #include "Eigen/SparseCore"
#include "Eigen/IterativeLinearSolvers"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TLine.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>


class ERecoSolver {

public:
  struct Params {
    enum Solver { kSVD = 1, kQR, kNormal, kInverse, kCOD, kConjugateGradient, kLeastSquaresConjugateGradient, kBiCGSTAB, kBiCGSTABguess };

    Solver algo_id;

    /// If using a FitBetween mode:
    /// 0: Ignore all bins outside range
    /// 1: Try to force bins to 0
    /// 2: Gaussian decay from target flux at closest kept bin.
    enum OutOfRangeModeEnum { kIgnore = 0, kZero, kGaussianDecay };

    /// How to deal with out of range
    OutOfRangeModeEnum OORMode;

    /// Varies rate of gaussian falloff if OORMode == kGaussianDecay
    double ExpDecayRate;

    /// If using an OOR mode:
    /// 0: Include both out of ranges
    /// 1: Only include out of range to the left of the range
    /// 2: Only include out of range to the right of the range
    enum OutOfRangeSideEnum { kBoth = 0, kLeft, kRight };

    /// Which side to fit
    OutOfRangeSideEnum OORSide;

    /// Down-weight contribution from the out of range bins;
    double OORFactor;

    double coeffMagLower;
    int LeastNCoeffs;

    bool FitBetweenFoundPeaks;
    std::pair<double, double> FitBetween;

    /// Number of ENu bins to merge
    int MergeENuBins;
    /// Number of OA position bins to merge
    int MergeOAPBins;

    /// Use a subset of the full input ranges described by this descriptor
    ///
    std::string OffAxisRangesDescriptor;

    std::string NominalOffAxisRangesDescriptor;

    std::pair<double, double> CurrentRange;

    std::vector<std::vector<std::pair<double, double>>> CurrentRangeSplit;

    std::pair<int, double> NominalFlux;

    std::string HConfigs;

    std::string WFile;

    enum Weighting { TotalFlux = 1, MaxFlux };

    Weighting WeightMethod;

    bool startCSequalreg;

    bool OrthogSolve;

  };
  Params fParams;

  static Params GetDefaultParams() {
    Params p;

    p.algo_id = Params::kSVD;
    p.OORMode = Params::kGaussianDecay;
    p.OORSide = Params::kLeft;
    p.OORFactor = 0.1;
    p.FitBetweenFoundPeaks = true;
    p.MergeENuBins = 0;
    p.MergeOAPBins = 0;
    p.OffAxisRangesDescriptor = "-1.45_37.55:0.1";
    p.NominalOffAxisRangesDescriptor = "-1.45_37.55:0.1";
    p.ExpDecayRate = 3;
    p.CurrentRange = {0, 350};
    p.CurrentRangeSplit = {};
    p.NominalFlux = {4, 293};
    p.HConfigs = "1,2,3,4,5";
    p.coeffMagLower = 0.0;
    p.LeastNCoeffs= 0;
    p.WFile = "";
    p.WeightMethod = Params::TotalFlux;
    p.startCSequalreg = false;
    p.OrthogSolve = false;

    return p;
  }

  void ComputeMatrix() {


  }

}
