#ifndef FLUXLINEARSOLVER_STANDALONE_HXX_SEEN
#define FLUXLINEARSOLVER_STANDALONE_HXX_SEEN

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

template<typename It>
int min_k(It first, It last, int k){
   k++; // k == 0 has to return smallest one element.
   auto cmp_it_values1 = [](It lt, It rt){ return *lt < *rt;};
   auto cmp_it_values2 = [](It lt, It rt){ return *lt <= *rt;};
   auto max_copy = std::min<long>(k, std::distance(first, last));
   auto start_it = first;
   std::advance(start_it, max_copy);
 
   // k++; // k == 0 has to return smallest one element.
    std::vector<It> k_smallest;
    k_smallest.reserve(k+1);
    for(auto it = first; it != start_it; ++it){
    	k_smallest.push_back(it);
    }
    std::stable_sort(k_smallest.begin(), k_smallest.end(), cmp_it_values1);
 
    for(auto it = start_it; it != last; ++it){
        if(k_smallest.empty() || *it <= *k_smallest.back()){
            auto insertion_point = std::lower_bound(k_smallest.begin(), k_smallest.end(), 
                                                    it, cmp_it_values2);
            k_smallest.insert(insertion_point, it);
            if(k_smallest.size() > k){
                k_smallest.pop_back(); // Remove the largest value
            }
        }
    }
    // std::cout << "Distance to vect : " << std::distance(first,k_smallest.back()) << std::endl;
    // int dist = std::distance(first,k_smallest.back());
    // std::cout << dist << std::endl;
    // std::cout << "*k_smallest.back() " << *k_smallest.back() << std::endl;
    // return k_smallest.back(); // The iterator to the min(k, n) smallest value 
    return (std::distance(first,k_smallest.back())); // The index of the min(k, n) smallest value 
    // return dist; 
}

#ifdef FLS_WRAP_IN_NAMESPACE
namespace fls {
#endif
// ************ Helper methods from StringParserUtility to make this standalone

template <typename T> inline T str2T(std::string const &str) {
  std::istringstream stream(str);
  T d;
  stream >> d;

  if (stream.fail()) {
    std::cerr << "[WARN]: Failed to parse string: " << str
              << " as requested type." << std::endl;
    return T();
  }

  return d;
}

template <typename T> inline std::string to_str(T const &inp) {
  std::stringstream stream("");
  stream << inp;
  return stream.str();
}
template <typename T>
inline std::vector<T> ParseToVect(std::string const &str, const char *del) {
  std::istringstream stream(str);
  std::string temp_string;
  std::vector<T> vals;

  while (std::getline(stream >> std::ws, temp_string, *del)) {
    if (temp_string.empty()) {
      continue;
    }
    vals.push_back(str2T<T>(temp_string));
  }
  return vals;
}

// Converts "5_10:1" into a vector containing: 5,6,7,8,9,10
inline std::vector<double> BuildDoubleList(std::string const &str) {
  std::vector<std::string> steps = ParseToVect<std::string>(str, ":");
  if (steps.size() != 2) {
    return ParseToVect<double>(str, ",");
  }
  double step = str2T<double>(steps[1]);

  std::vector<double> range = ParseToVect<double>(steps[0], "_");
  if (steps.size() != 2) {
    std::cout
        << "[ERROR]: When attempting to parse bin range descriptor: \" " << str
        << "\", couldn't determine range. Expect form: <bin1low>_<binXUp>:step"
        << std::endl;
    exit(1);
  }

  int nsteps = (range[1] - range[0]) / step;

  std::vector<double> rtn;
  for (int step_it = 0; step_it <= nsteps; ++step_it) {
    rtn.push_back(range[0] + step * step_it);
  }
  return rtn;
}
inline std::vector<std::pair<double, double>>
BuildRangesList(std::string const &str) {
  std::vector<std::string> listDescriptor = ParseToVect<std::string>(str, ",");
  std::vector<std::pair<double, double>> RangesList;

  for (size_t l_it = 0; l_it < listDescriptor.size(); ++l_it) {
    // If this includes a range to build
    if (listDescriptor[l_it].find("_") != std::string::npos) {
      std::vector<std::string> rangeDescriptor =
          ParseToVect<std::string>(listDescriptor[l_it], ":");

      if (rangeDescriptor.size() != 2) {
        std::cout
            << "[ERROR]: Range descriptor: \"" << str
            << "\" contained bad descriptor: \"" << listDescriptor[l_it]
            << "\", expected <RangeCenterLow>_<RangeCenterHigh>:<RangeWidths>."
            << std::endl;
        exit(0);
      }

      std::vector<double> rangeCenters = BuildDoubleList(listDescriptor[l_it]);
      double width = str2T<double>(rangeDescriptor[1]);

      for (size_t sp_it = 0; sp_it < rangeCenters.size(); ++sp_it) {
        RangesList.push_back(
            std::make_pair(rangeCenters[sp_it] - (width / 2.0),
                           rangeCenters[sp_it] + (width / 2.0)));
      }

    } else {
      std::vector<double> rangeDescriptor =
          ParseToVect<double>(listDescriptor[l_it], ":");
      if (rangeDescriptor.size() != 2) {
        std::cout << "[ERROR]: Range descriptor: \"" << str
                  << "\" contained bad descriptor: \"" << listDescriptor[l_it]
                  << "\", expected <RangeCenter>:<RangeWidth>." << std::endl;
        exit(0);
      }
      RangesList.push_back(
          std::make_pair(rangeDescriptor[0] - (rangeDescriptor[1] / 2.0),
                         rangeDescriptor[0] + (rangeDescriptor[1] / 2.0)));
    }
  }

  for (size_t range_it = 1; range_it < RangesList.size(); ++range_it) {
    if ((RangesList[range_it - 1].second - RangesList[range_it].first) > 1E-5) {
      std::cout << "[ERROR]: Range #" << range_it << " = {"
                << RangesList[range_it].first << " -- "
                << RangesList[range_it].second << "}, but #" << (range_it - 1)
                << " = {" << RangesList[range_it - 1].first << " -- "
                << RangesList[range_it - 1].second << "}." << std::endl;
      exit(1);
    }
  }
  return RangesList;
}

inline std::vector<int>
BuildHConfsList(std::string const &str) {
  std::vector<int> HClist = ParseToVect<int>(str, ",");

  if (! HClist.size() ) {
        std::cout
            << "[ERROR]: HConfdescriptor: \"" << str
            << "\" contained bad descriptor."
            << " Expected integers between 1 and 5 separated by commas."
            << std::endl;
        exit(0);
      }
  else for ( int l : HClist ) {
    if ( l > 5 || l < 0 ) {
        std::cout
            << "[ERROR]: HConfdescriptor: \"" << str
            << "\" contained bad descriptor: \"" << l
            << "\", expected integers between 1 and 5 separated by commas."
            << std::endl;
        exit(0);
      }
  }
  return HClist;
}

// ************ End helper methods

// *************** Helper methods from ROOTUtility to make this standalone

inline TFile *CheckOpenFile(std::string const &fname, char const *opts = "") {
  TFile *inpF = new TFile(fname.c_str(), opts);
  if (!inpF || !inpF->IsOpen()) {
    std::cout << "[ERROR]: Couldn't open input file: " << fname << std::endl;
    exit(1);
  }
  return inpF;
}

template <class TH>
inline std::unique_ptr<TH> GetHistogram(TFile *f, std::string const &fhname,
                                        bool no_throw = false) {
  TH *fh = dynamic_cast<TH *>(f->Get(fhname.c_str()));

  if (!fh) {
    if (no_throw) {
      return std::unique_ptr<TH>(nullptr);
    }
    std::cout << "[ERROR]: Couldn't get TH: " << fhname
              << " from input file: " << f->GetName() << std::endl;
    exit(1);
  }

  std::unique_ptr<TH> inpH =
      std::unique_ptr<TH>(static_cast<TH *>(fh->Clone()));

  inpH->SetDirectory(nullptr);
  return inpH;
}

template <class TH>
inline std::unique_ptr<TH> GetHistogram(std::string const &fname,
                                        std::string const &hname,
                                        bool no_throw = false) {
  TDirectory *ogDir = gDirectory;

  TFile *inpF = CheckOpenFile(fname);

  std::unique_ptr<TH> h =
#ifdef FLS_WRAP_IN_NAMESPACE
      fls::
#endif
          GetHistogram<TH>(inpF, hname, no_throw);

  inpF->Close();
  delete inpF;

  if (ogDir) {
    ogDir->cd();
  }

  return h;
}

inline std::pair<Int_t, Int_t>
GetProjectionBinRange(std::pair<double, double> ValRange, TAxis *axis) {
  Int_t low_bin = axis->FindFixBin(ValRange.first);
  if (fabs(axis->GetBinUpEdge(low_bin) - ValRange.first) < 1E-5) {
    low_bin += 1;
  }
  Int_t high_bin = axis->FindFixBin(ValRange.second);
  if (fabs(axis->GetBinLowEdge(high_bin) - ValRange.second) < 1E-5) {
    high_bin -= 1;
  }

  if (fabs(axis->GetBinLowEdge(low_bin) - ValRange.first) > 1E-5) {
    std::cout << "[ERROR]: Chose first projection bin = " << low_bin
              << ", with low edge = " << axis->GetBinLowEdge(low_bin)
              << ", but the starting range was requested as " << ValRange.first
              << std::endl;
    exit(1);
  }
  if (fabs(axis->GetBinUpEdge(high_bin) - ValRange.second) > 1E-5) {
    std::cout << "[ERROR]: Chose last projection bin = " << high_bin
              << ", with up edge = " << axis->GetBinLowEdge(high_bin)
              << ", but the ending range was requested as " << ValRange.second
              << std::endl;
    exit(1);
  }

  if (low_bin == 0) {
    std::cout << "[ERROR]: Low bin is underflow bin, 2D flux is not adequate "
                 "for this splitting scheme"
              << std::endl;
    exit(1);
  }
  if (high_bin == (axis->GetNbins() + 1)) {
    std::cout << "[ERROR]: High bin is overflow bin, 2D flux is not adequate "
                 "for this splitting scheme"
              << std::endl;
    exit(1);
  }
  return std::make_pair(low_bin, high_bin);
}

inline std::vector<std::unique_ptr<TH1>>
MergeSplitTH2(std::unique_ptr<TH2> &t2, bool AlongY,
              std::vector<std::pair<double, double>> const &Vals) {

  std::vector<std::unique_ptr<TH1>> split;

    for (std::pair<double, double> const &v : Vals) {
      std::pair<Int_t, Int_t> binr =
          GetProjectionBinRange(v, (AlongY ? t2->GetYaxis() : t2->GetXaxis()));

      split.emplace_back(
          AlongY ? t2->ProjectionX(
                       (to_str(v.first) + "_to_" + to_str(v.second)).c_str(),
                       binr.first, binr.second)
                 : t2->ProjectionY(
                       (to_str(v.first) + "_to_" + to_str(v.second)).c_str(),
                       binr.first, binr.second));
      split.back()->Scale(1.0 / double(binr.second - binr.first + 1));
      split.back()->SetDirectory(NULL);
    }

  return split;
}

inline int FindTH1Peaks(TH1 const *flux, int &left, int &right, int n) {

  std::unique_ptr<TH1D> temp = std::unique_ptr<TH1D>(
      static_cast<TH1D *>(flux->Clone("peakfindingtemp")));
  temp->SetDirectory(nullptr);
  temp->Smooth(10);

  double threshold = (temp->Integral()) / (5 * (temp->GetNbinsX()));

  int nfound = 0;
  double content[3] = {0};

  for (int bin_ind = temp->GetNbinsX(); bin_ind > 0 && nfound < n; bin_ind--) {
    content[2] = temp->GetBinContent(bin_ind - 1);
    if ((content[0] < content[1]) && (content[1] > content[2]) &&
        (content[1] > threshold)) {
      if (nfound == 0) {
        right = bin_ind;
      }
      if (nfound == (n - 1)) {
        left = bin_ind;
      }
      nfound++;
    }
    content[0] = content[1];
    content[1] = content[2];
  }

  return nfound;
}

inline void FillHistFromEigenVector(TH1 *rh, Eigen::VectorXd const &vals,
                                    size_t bin_offset = 0) {
  Int_t dim = rh->GetDimension();
  if (dim == 1) {
    size_t idx = 0;
    std::cout << "Filling histogram with " << rh->GetXaxis()->GetNbins()
              << " bins from vector with " << vals.rows() << " rows."
              << std::endl;
    for (Int_t x_it = bin_offset; x_it < rh->GetXaxis()->GetNbins(); ++x_it) {
      double v = (idx >= vals.rows()) ? 0 : vals(idx);
      rh->SetBinContent(x_it + 1, v);
      rh->SetBinError(x_it + 1, 0);
      idx++;
    }
    // Reset flow bins
    rh->SetBinContent(0, 0);
    rh->SetBinError(0, 0);
    rh->SetBinContent(rh->GetXaxis()->GetNbins() + 1, 0);
    rh->SetBinError(rh->GetXaxis()->GetNbins() + 1, 0);
    return;
  }
  std::cout << "[ERROR]: FillHistFromstdvector cannot handle THND where N = "
            << dim << std::endl;
  exit(1);
}

// ****************** End helper methods

class FluxLinearSolver {

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

    return p;
  }

  Eigen::MatrixXd FluxMatrix_Full;
  Eigen::MatrixXd FluxMatrix_zSlice;
  Eigen::MatrixXd FluxMatrix_All; 
  Eigen::MatrixXd FluxMatrix_Solve;
  Eigen::MatrixXd FluxMatrix_Reduced;
  Eigen::VectorXd Target;

  Eigen::MatrixXd FDWeights;

  Eigen::VectorXd last_solution;
  bool soln_set = false;
  std::vector<double> coeffvec;

  std::unique_ptr<TH1> FDFlux_unosc;
  std::unique_ptr<TH1> FDFlux_osc;

  std::unique_ptr<TTree> BCTree;
  
  Double_t RegParam;
  Double_t BCParam;
  Double_t CSNorm;
  Double_t StabilityFactor;

  // Int_t NumBeamConfigs = 0;
  // Int_t TotalNumBeamConfigs = 0;
  size_t NCoefficients;
  std::vector<size_t> nZbins;
  std::vector<std::vector<size_t>> AllZbins;
  std::vector<std::vector<size_t>> AllOAbins;
  std::vector<Int_t> OAbinsperhist;
  size_t NomRegFluxFirst = 0, NomRegFluxLast = 0;
  size_t low_offset, FitIdxLow, FitIdxHigh;
  std::vector<bool> UseFluxesOld; 
  std::vector<bool> UseFluxesNew; 
  bool ApplyWeightings = false;

  void
  Initialize(Params const &p,
	     int &ncoeffs,
             std::pair<std::string, std::string> NDFluxDescriptor = {"", ""},
             std::pair<std::string, std::string> FDFluxDescriptor = {"", ""},
             std::pair<std::string, std::string> NDBeamConfDescriptor = {"", ""},
             bool FDIsOsc = false) {

    std::pair<std::string, std::vector<std::string>> vectNDFluxDescriptor;
    vectNDFluxDescriptor.first = NDFluxDescriptor.first;
    vectNDFluxDescriptor.second.emplace_back(NDFluxDescriptor.second);

    Initialize(p, ncoeffs, vectNDFluxDescriptor, FDFluxDescriptor,
		NDBeamConfDescriptor, FDIsOsc);

  }

  void
  Initialize(Params const &p,
	     int &ncoeffs,
             std::pair<std::string, std::vector<std::string>> NDFluxDescriptor = {"", {""} },
             std::pair<std::string, std::string> FDFluxDescriptor = {"", ""},
             std::pair<std::string, std::string> NDBeamConfDescriptor = {"", ""},
             bool FDIsOsc = false) {

    FluxMatrix_Full = Eigen::MatrixXd::Zero(0, 0);
    FluxMatrix_All = Eigen::MatrixXd::Zero(0, 0);

    fParams = p;

    std::vector<std::unique_ptr<TH3>> Flux3DList;
    for (size_t hist_it=0; hist_it < NDFluxDescriptor.second.size(); hist_it++) {
      if (NDFluxDescriptor.first.size() && NDFluxDescriptor.second.size()) {

        std::unique_ptr<TH3> Flux3D =
            GetHistogram<TH3>(NDFluxDescriptor.first, NDFluxDescriptor.second[hist_it]);

        if (!Flux3D) {
          std::cout << "[ERROR]: Found no input flux with name: \"" 
                    << NDFluxDescriptor.first << "\" in file: \"" 
                    << NDFluxDescriptor.second[hist_it] << "\"." << std::endl;
          throw;
        }

	Flux3DList.emplace_back(std::move(Flux3D));
      }
    } 
    SetNDFluxes(std::move(Flux3DList));
    // end ND setup

    if (FDFluxDescriptor.first.size() && FDFluxDescriptor.second.size()) {

      std::unique_ptr<TH1> FDFlux =
          GetHistogram<TH1>(FDFluxDescriptor.first, FDFluxDescriptor.second);
      if (!FDIsOsc) {
        SetFDFluxUnOsc(FDFlux.get());
      } else {
        SetFDFluxOsc(FDFlux.get());
        BuildTargetFlux();
      }
    }

    if (fParams.WFile.size()) {
      std::cout << "[INFO]: Using FD Flux Reg Weighting" << std::endl;
      BuildWeights(NDFluxDescriptor.second);
    }

    ncoeffs = FluxMatrix_Solve.cols();

    /*TFile *BCfile = CheckOpenFile(NDFluxDescriptor.first);
    TTree *BCtree = (TTree*)BCfile->Get((NDBeamConfDescriptor.first).c_str());
    if (!BCtree) {
      std::cout << "Couldn't find config tree: " << (NDBeamConfDescriptor.first).c_str() << std::endl;
      NumBeamConfigs=0;
      std::cout << "Number of additional Beam Configs used = " << NumBeamConfigs << std::endl;
      return;
    }
    BCtree = (TTree*)BCfile->Get((NDBeamConfDescriptor.first).c_str());
    BCtree->SetBranchAddress("NumBeamConfigs",&NumBeamConfigs); 
    for (int BC_it = 0; BC_it < BCtree->GetEntries(); BC_it++) {
	BCtree->GetEntry(BC_it);
	TotalNumBeamConfigs += NumBeamConfigs;
    }
    std::cout << "Number of additional beam configs in file = " << TotalNumBeamConfigs << std::endl;
    */
  }

  void SetNDFluxes(std::vector<std::unique_ptr<TH3>> const &NDFluxes, bool ApplyXRanges = true) {

    std::vector<std::pair<double, double>> XRanges;
    if (ApplyXRanges && fParams.OffAxisRangesDescriptor.size()) {
      XRanges = BuildRangesList(fParams.OffAxisRangesDescriptor);
    }

    std::vector<std::pair<double, double>> NomXRanges;
    if (ApplyXRanges && fParams.NominalOffAxisRangesDescriptor.size()) {
      NomXRanges = BuildRangesList(fParams.NominalOffAxisRangesDescriptor);
    }

    std::vector<int> Confs;
    if (fParams.HConfigs.size()) {
      Confs = BuildHConfsList(fParams.HConfigs);
      // consider removing (+ BuildHConfsList function) 
    }

    // std::cout << NDFluxes[0]->GetNbinsZ() << std::endl;
    std::vector<Eigen::MatrixXd> NDMatrices;

    for (size_t Hist3D_it = 0; Hist3D_it < NDFluxes.size(); Hist3D_it++ ) {
      std::unique_ptr<TH3> Flux3D(static_cast<TH3 *>(NDFluxes[Hist3D_it]->Clone()));
      Flux3D->SetDirectory(nullptr);

      if ( ! fParams.CurrentRangeSplit.size() ) {
	std::vector<std::pair<double, double>> tmpvec{{fParams.CurrentRange.first, fParams.CurrentRange.second}};
	// fParams.CurrentRangeSplit.emplace_back(tmpvec);
	fParams.CurrentRangeSplit.assign(NDFluxes.size(), tmpvec);
      }

      std::vector<std::pair<double, double>> HistCurrents = fParams.CurrentRangeSplit[Hist3D_it];
      std::vector<size_t> HistZbins;
      for (size_t zbi_it = 1; zbi_it <= Flux3D->GetZaxis()->GetNbins(); zbi_it++) {
        // if high edge is higher than first val and low edge is lower than second val
        for (int ranges_it = 0; ranges_it < HistCurrents.size(); ranges_it++) {
          if ( Flux3D->GetZaxis()->GetBinUpEdge(zbi_it) > HistCurrents[ranges_it].first &&
		 Flux3D->GetZaxis()->GetBinLowEdge(zbi_it) < HistCurrents[ranges_it].second ) {
	    HistZbins.emplace_back(zbi_it);
	  }
	}
      }
      AllZbins.emplace_back(HistZbins);

      int NominalZbin = Flux3D->GetZaxis()->FindFixBin( fParams.NominalFlux.second );

      int zBins = HistZbins.size(); 
      nZbins.emplace_back(zBins);
      std::vector<size_t> nOAbins;
      
/*
      int lowCurrentBin;
      if ( fParams.CurrentRangeSplit.size() ) {
	std::cout << "Using split currents : " << std::endl;
	std::cout << "Hist " << Hist3D_it + 1 << " with " << fParams.CurrentRangeSplit[Hist3D_it].first << "," << fParams.CurrentRangeSplit[Hist3D_it].second << std::endl;
        lowCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRangeSplit[Hist3D_it].first );
      } else {
        lowCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRange.first );
      }
      if ( lowCurrentBin == 0 ) {
	lowCurrentBin = 1; // If below bin range, set to minimum non-underflow bin
      }

      int NominalZbin = Flux3D->GetZaxis()->FindFixBin( fParams.NominalFlux.second );

      int highCurrentBin;
      if ( fParams.CurrentRangeSplit.size() ) {
        highCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRangeSplit[Hist3D_it].second );
      } else {
        highCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRange.second );
      }
      if ( highCurrentBin == (Flux3D->GetZaxis()->GetNbins() + 1) ) {
	highCurrentBin = Flux3D->GetZaxis()->GetNbins(); // If above bin range, set to maximum non-overflow bin
      }
      int zBins = ( highCurrentBin + 1 ) - lowCurrentBin;
      nZbins.emplace_back(zBins);
      std::vector<size_t> nOAbins;
      // std::cout << " lowCurrentBin : " << lowCurrentBin << std::endl;
      // std::cout << " NominalZbin : " << NominalZbin << std::endl;
      // std::cout << " highCurrentBin : " << highCurrentBin << std::endl; 
      // std::cout << " zBins : " << zBins << std::endl; 
*/

      // for (int z = lowCurrentBin; z <= highCurrentBin; z++) {
      // }

      for ( size_t z : HistZbins ) {
        Flux3D->GetZaxis()->SetRange(z,z);
        // std::unique_ptr<TH2> Flux2D = Flux3D->Project3D("yx");
        TH2 *projectedFlux = (TH2*)Flux3D->Project3D("yx");
      	std::unique_ptr<TH2> Flux2D(static_cast<TH2 *>(projectedFlux->Clone()));
      	Flux2D->SetDirectory(nullptr);

        if (fParams.MergeENuBins && fParams.MergeOAPBins) {
          Flux2D->Rebin2D(fParams.MergeENuBins, fParams.MergeOAPBins);
          Flux2D->Scale(1.0 / double(fParams.MergeENuBins * fParams.MergeOAPBins));
        } else if (fParams.MergeENuBins) {
          Flux2D->RebinX(fParams.MergeENuBins);
          Flux2D->Scale(1.0 / double(fParams.MergeENuBins));
        } else if (fParams.MergeOAPBins) {
          Flux2D->RebinY(fParams.MergeOAPBins);
          Flux2D->Scale(1.0 / double(fParams.MergeOAPBins));
        }

    	std::vector<std::pair<double, double>> UseXRanges = XRanges;
        if ( (Hist3D_it == (fParams.NominalFlux.first - 1)) && (z == NominalZbin) && (NomXRanges.size()) ) {
	  UseXRanges=NomXRanges;
	}
	if (UseXRanges.size()) { // Use a subset of (possibly merged) off-axis slices
          std::vector<std::unique_ptr<TH1>> FluxSlices =
              MergeSplitTH2(Flux2D, true, UseXRanges);
          if (!FluxSlices.size()) {
            std::cout << "[ERROR]: Found no input fluxes." << std::endl;
            exit(1);
          }
          // extra rows corresponding to NColumns used for regularization if
          // enabled
          FluxMatrix_zSlice = Eigen::MatrixXd::Zero(
              FluxSlices.front()->GetXaxis()->GetNbins(), FluxSlices.size());
          size_t col_it = 0;
          for (std::unique_ptr<TH1> &f : FluxSlices) {
            for (Int_t bi_it = 0; bi_it < f->GetXaxis()->GetNbins(); ++bi_it) {
              FluxMatrix_zSlice(bi_it, col_it) = f->GetBinContent(bi_it + 1);
            }
            col_it++;
          }
	  // std::cout << "XBins : " << FluxSlices.front()->GetXaxis()->GetNbins() << std::endl;
        } else { // Use the entire set of fluxes
          // extra rows corresponding to NColumns used for regularization if
          // enabled
          FluxMatrix_zSlice = Eigen::MatrixXd::Zero(Flux2D->GetXaxis()->GetNbins(),
                                                Flux2D->GetYaxis()->GetNbins());
          for (Int_t oabi_it = 0; oabi_it < Flux2D->GetYaxis()->GetNbins();
               ++oabi_it) {
            for (Int_t ebi_it = 0; ebi_it < Flux2D->GetXaxis()->GetNbins();
                 ++ebi_it) {
              FluxMatrix_zSlice(ebi_it, oabi_it) =
                  Flux2D->GetBinContent(ebi_it + 1, oabi_it + 1);
            }
          }
        }
	NDMatrices.emplace_back(FluxMatrix_zSlice);
	nOAbins.emplace_back(FluxMatrix_zSlice.cols());
	std::cout << "nOAbins[z] : " << FluxMatrix_zSlice.cols() << std::endl;
	// Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	// std::cout << FluxMatrix_zSlice.format(CleanFmt) << std::endl;
	
	// for (size_t check_it = 0; check_it < FluxMatrix_zSlice.rows(); check_it++) {
	//     std::cout << "\n\nRow : " << check_it << "\nFluxRow:\n " << FluxMatrix_zSlice.row(check_it) << std::endl;
	// }
      }

      AllOAbins.emplace_back(nOAbins);

      if ( Hist3D_it == (fParams.NominalFlux.first - 1) ) {
	// runs through all previous 3D histograms
        for (size_t prevhist_it = 0; prevhist_it < Hist3D_it; prevhist_it++) {
	  std::cout << " nZbins["<<prevhist_it<<"] : " <<  nZbins[prevhist_it] << std::endl; 
    	  for (size_t OAbins_it = 0; OAbins_it < nZbins[prevhist_it]; OAbins_it++) {
  	    NomRegFluxFirst += AllOAbins[prevhist_it][OAbins_it];
	    // NomRegFluxFirst += nOAbins[OAbins_it];
	    std::cout << "NomRegFluxFirst (counting) : " << NomRegFluxFirst << std::endl;
	  }
	}
	std::cout << " nZbins[Hist3D_it] : " <<  nZbins[Hist3D_it] << std::endl; 

	// Get iterator to nominal z bin
    	std::vector<size_t>::iterator it = std::find( AllZbins[Hist3D_it].begin(), AllZbins[Hist3D_it].end(), NominalZbin );
	if (it == AllZbins[Hist3D_it].end()) {
	  std::cout << "[ERROR] : Nominal Z bin not in curent range passed to fit" << std::endl;
          exit(1);
	}
	// Get nom index as distance from start to iterator
	int index = std::distance(AllZbins[Hist3D_it].begin(), it);
	// runs through all previous z slices (other current settings) in this histogram 
        for (size_t OAbins_it = 0; OAbins_it < index; OAbins_it++) {
	  NomRegFluxFirst += nOAbins[OAbins_it];
	}
	NomRegFluxFirst += 1; 
	NomRegFluxLast += NomRegFluxFirst + nOAbins[index] - 1; // adds the OA fluxes for this setting to obtain NomRegFluxLast

	std::cout << "NomRegFluxFirst = " << NomRegFluxFirst << std::endl;
	std::cout << "NomRegFluxLast = " << NomRegFluxLast<< std::endl;
      }
    }
    Int_t NDrows = NDMatrices[0].rows();
    // Int_t NDcols = NDMatrices[0].cols();
    // std::cout << "NDcols : " << NDcols << std::endl;
    Int_t FullMcols = 0;
    // std::cout << " AllOAbins.size() : " << AllOAbins.size() << std::endl;
    // std::cout << " AllOAbins[0].size() : " << AllOAbins[0].size() << std::endl;
    // auto v1 = AllOAbins[0];
    for (auto v1 : AllOAbins) {
      for (auto v2 : v1) {
        FullMcols += v2;
      }
    }

    OAbinsperhist.assign(AllOAbins.size(), 0);
    for (size_t v1 = 0; v1 < AllOAbins.size(); v1++) {
      for (size_t v2 = 0; v2 < AllOAbins[v1].size(); v2++) {
        OAbinsperhist[v1] += AllOAbins[v1][v2];
      }
    }
    // std::cout << "FullMcols : " << FullMcols << std::endl;
    // std::cout << "NDMatrices.size(): " << NDMatrices.size() << std::endl;

    // std::cout << "NDrows : " << NDrows << std::endl;
    // std::cout << "NDcols : " << NDcols << std::endl;

    // std::cout << "NDMatrices.size : " << NDMatrices.size() << std::endl;

    // FluxMatrix_Full = Eigen::MatrixXd(NDrows, NDcols*NDMatrices.size());
    FluxMatrix_Full = Eigen::MatrixXd(NDrows, FullMcols);
    for (size_t n = 0; n < NDMatrices.size(); n++) {
      // Int_t prevcols = all NDcols for m < n
      Int_t prevcols = 0;
      for (int prev_it = 0; prev_it < n; prev_it++) {
        prevcols += NDMatrices[prev_it].cols();
      }
      FluxMatrix_Full.block(0, 0 + prevcols, NDrows, NDMatrices[n].cols()) = NDMatrices[n];
    }
    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FluxMatrix_Full.format(CleanFmt) << std::endl;
    NCoefficients = FluxMatrix_Full.cols();

    if ( NCoefficients < fParams.LeastNCoeffs ) {
      std::cout << "[ERROR] : -CNL = " << fParams.LeastNCoeffs << " larger than NFluxes : " << NCoefficients << std::endl;
      exit(1);
    }

  }

  void OscillateFDFlux(std::array<double, 6> OscParameters = {},
                       std::pair<int, int> OscChannel = {14, 14},
                       double DipAngle_degrees = 5.8) {

    BargerPropagator bp;

    int OscFrom = (OscChannel.first / 2) + (OscChannel.first > 0 ? -5 : 5);
    int OscTo = (OscChannel.second / 2) + (OscChannel.second > 0 ? -5 : 5);

    std::cout << "Osc from " << OscFrom << ", Osc to " << OscTo << std::endl;

    double LengthParam = cos((90.0 + DipAngle_degrees) * (asin(1) / 90.0));

    // Oscillate the flux
    Int_t NEBins = FDFlux_unosc->GetXaxis()->GetNbins();
    for (Int_t bi_it = 0; bi_it < NEBins; ++bi_it) {
      double ENu_GeV = FDFlux_unosc->GetXaxis()->GetBinCenter(bi_it + 1);
      bp.SetMNS(OscParameters[0], OscParameters[1], OscParameters[2],
                OscParameters[3], OscParameters[4], OscParameters[5], ENu_GeV,
                true, OscFrom);
      bp.DefinePath(LengthParam, 0);
      bp.propagate(OscTo);

      FDFlux_osc->SetBinContent(bi_it + 1,
                                bp.GetProb(OscFrom, OscTo) *
                                    FDFlux_unosc->GetBinContent(bi_it + 1));
    }

    BuildTargetFlux();
  }

  void SetFDFluxUnOsc(TH1 *const FDFlux) {
    FDFlux_unosc = std::unique_ptr<TH1>(static_cast<TH1 *>(FDFlux->Clone()));
    FDFlux_unosc->SetDirectory(nullptr);

    if (fParams.MergeENuBins) {
      FDFlux_unosc->Rebin(fParams.MergeENuBins);
      FDFlux_unosc->Scale(1.0 / double(fParams.MergeENuBins));
    }

    FDFlux_osc =
        std::unique_ptr<TH1>(static_cast<TH1 *>(FDFlux_unosc->Clone()));
    FDFlux_osc->SetDirectory(nullptr);
    FDFlux_osc->Reset();
  }

  void SetFDFluxOsc(TH1 *const FDFlux) {
    FDFlux_osc = std::unique_ptr<TH1>(static_cast<TH1 *>(FDFlux->Clone()));
    FDFlux_osc->SetDirectory(nullptr);

    if (fParams.MergeENuBins) {
      FDFlux_osc->Rebin(fParams.MergeENuBins);
      FDFlux_osc->Scale(1.0 / double(fParams.MergeENuBins));
    }

    FDFlux_unosc = nullptr;
  }

  TH1 *GetFDFluxToOsc() {
    Int_t NEBins = FDFlux_unosc->GetXaxis()->GetNbins();

    for (Int_t bi_it = 0; bi_it < NEBins; ++bi_it) {
      FDFlux_osc->SetBinContent(bi_it + 1,
                                FDFlux_unosc->GetBinContent(bi_it + 1));
    }

    return FDFlux_osc.get();
  }

  void BuildTargetFlux() {
    int FitBinLow = 1;
    int FitBinHigh = FDFlux_osc->GetXaxis()->GetNbins();
    low_offset = 0;

    if (fParams.FitBetweenFoundPeaks) {
      int nfound = FindTH1Peaks(FDFlux_osc.get(), FitBinLow, FitBinHigh, 3);
      if (nfound != 3) {
        std::cout << "[ERROR]: Failed to find the expected number of "
                     "peaks, "
                  << std::endl;
        throw;
      }
      fParams.FitBetween.first =
          FDFlux_osc->GetXaxis()->GetBinLowEdge(FitBinLow);
      fParams.FitBetween.second =
          FDFlux_osc->GetXaxis()->GetBinUpEdge(FitBinHigh);

      std::cout << "[INFO]: Found flux peaks @ " << FitBinLow << " = "
                << fParams.FitBetween.first << ", and " << FitBinHigh << " = "
                << fParams.FitBetween.second << std::endl;
    } else {
      if (fParams.FitBetween.first == 0xdeadbeef) {
        FitBinLow = 1;
        FitBinHigh = FDFlux_osc->GetXaxis()->GetNbins();
      } else {
        FitBinLow =
            FDFlux_osc->GetXaxis()->FindFixBin(fParams.FitBetween.first);
        FitBinHigh =
            FDFlux_osc->GetXaxis()->FindFixBin(fParams.FitBetween.second);
      }
    }

    FitIdxLow = FitBinLow - 1;
    FitIdxHigh = FitBinHigh - 1;

    if (fParams.OORMode ==
        Params::kIgnore) { // Throw away all info outside of fit region

      low_offset = FitIdxLow;

      size_t NEBinRows = FitBinHigh - FitBinLow;
      // Include space for regulaization constraint
      FluxMatrix_Solve =
          Eigen::MatrixXd::Zero(NEBinRows + NCoefficients, NCoefficients);

      FluxMatrix_Solve.topRows(NEBinRows) =
          FluxMatrix_Full.topRows(FitBinHigh - 1).bottomRows(NEBinRows);

      Target = Eigen::VectorXd::Zero(NEBinRows + NCoefficients);
      Int_t t_it = 0;
      for (Int_t bi_it = FitBinLow; bi_it < (FitBinHigh + 1); ++bi_it) {
        Target(t_it++) = FDFlux_osc->GetBinContent(bi_it);
      }

      // We don't have any out of ranges to build
      return;

    } else {

      // Set up FluxMatrix_Solve
      if (fParams.OORSide == Params::kBoth) {

        size_t NEBinRows = FluxMatrix_Full.rows();
        size_t NLowRows = FitBinLow - 1;
        size_t NHighRows = NEBinRows - (FitBinHigh - 1);

        FluxMatrix_Solve =
            Eigen::MatrixXd::Zero(NEBinRows + NCoefficients, NCoefficients);

        FluxMatrix_Solve.topRows(NEBinRows) = FluxMatrix_Full;

        // Setup OORFactor
        FluxMatrix_Solve.topRows(NLowRows).array() *= fParams.OORFactor;
        FluxMatrix_Solve.topRows(NEBinRows).bottomRows(NHighRows).array() *=
            fParams.OORFactor;

      } else if (fParams.OORSide == Params::kLeft) {

        size_t NEBinRows = (FitBinHigh - 1);
        size_t NLowRows = FitBinLow - 1;

        FluxMatrix_Solve =
            Eigen::MatrixXd::Zero(NEBinRows + NCoefficients, NCoefficients);

        FluxMatrix_Solve.topRows(NEBinRows) =
            FluxMatrix_Full.topRows(NEBinRows);

        // Setup OORFactor
        FluxMatrix_Solve.topRows(NLowRows).array() *= fParams.OORFactor;

      } else if (fParams.OORSide == Params::kRight) {
        size_t NEBinRows = FluxMatrix_Full.rows() - (FitBinLow - 1);
        size_t NHighRows = (FitBinHigh - 1);

        low_offset = FitIdxLow;

        FluxMatrix_Solve =
            Eigen::MatrixXd::Zero(NEBinRows + NCoefficients, NCoefficients);

        FluxMatrix_Solve.topRows(NEBinRows) =
            FluxMatrix_Full.bottomRows(NEBinRows);

        // Setup OORFactor
        FluxMatrix_Solve.topRows(NEBinRows).bottomRows(NHighRows).array() *=
            fParams.OORFactor;
      }

      Target = Eigen::VectorXd::Zero(FluxMatrix_Solve.rows());

      size_t t_it = 0;
      if ((fParams.OORSide == Params::kBoth) ||
          (fParams.OORSide == Params::kLeft)) {
        for (Int_t bi_it = 1; bi_it < FitBinLow; ++bi_it) {

          double oor_target = 0;
          double enu_first_counted_bin =
              FDFlux_osc->GetXaxis()->GetBinCenter(FitBinLow);
          double enu = FDFlux_osc->GetXaxis()->GetBinCenter(bi_it);
          double content_first_counted_bin =
              FDFlux_osc->GetBinContent(FitBinLow);
          double enu_bottom_bin = FDFlux_osc->GetXaxis()->GetBinCenter(1);
          double sigma5_range = enu_first_counted_bin - enu_bottom_bin;

          if (fParams.OORMode == Params::kGaussianDecay) {
            oor_target =
                content_first_counted_bin *
                exp(-fParams.ExpDecayRate * (enu_first_counted_bin - enu) *
                    (enu_first_counted_bin - enu) /
                    (sigma5_range * sigma5_range));
          }

          Target(t_it++) = oor_target * fParams.OORFactor;
        }
      }
      for (Int_t bi_it = FitBinLow; bi_it < (FitBinHigh + 1); ++bi_it) {
        Target(t_it++) = FDFlux_osc->GetBinContent(bi_it);
      }
      if ((fParams.OORSide == Params::kBoth) ||
          (fParams.OORSide == Params::kRight)) {
        // Build the target above the fit region
        for (Int_t bi_it = (FitBinHigh + 1);
             bi_it < FDFlux_osc->GetXaxis()->GetNbins() + 1; ++bi_it) {

          double oor_target = 0;
          double enu_last_counted_bin =
              FDFlux_osc->GetXaxis()->GetBinCenter(FitBinHigh);
          double enu = FDFlux_osc->GetXaxis()->GetBinCenter(bi_it);
          double content_last_counted_bin =
              FDFlux_osc->GetBinContent(FitBinHigh);
          double enu_top_bin = FDFlux_osc->GetXaxis()->GetBinCenter(
              FDFlux_osc->GetXaxis()->GetNbins());
          double sigma5_range = enu_top_bin - enu_last_counted_bin;

          if (fParams.OORMode == Params::kGaussianDecay) {
            oor_target =
                content_last_counted_bin *
                exp(-fParams.ExpDecayRate * (enu - enu_last_counted_bin) *
                    (enu - enu_last_counted_bin) /
                    (sigma5_range * sigma5_range));
          }
          Target(t_it++) = oor_target * fParams.OORFactor;
        }
      }
    }
  }

  Eigen::VectorXd const &Solve(double reg_param, double BC_param, double &res_norm,
                               double &soln_norm) {

   bool use_reg = reg_param > 0;
   size_t NFluxes = FluxMatrix_Solve.cols();
   size_t NEqs = FluxMatrix_Solve.rows();
   size_t NBins = NEqs - NFluxes;
   //double NomRegFluxFirst, NomRegFluxLast;
   UseFluxesOld.assign(NFluxes,true);
   UseFluxesNew.assign(NFluxes,false);
   int loops = 0;


   // std::cout << "[INFO]: Solving with " << NBins << " energy bins." << std::endl;
   std::cout << "[INFO]: Solving with " << NFluxes << " fluxes." << std::endl;
   // std::cout << "\n[INFO]: Solving with " << NumBeamConfigs << " extra configs in file." << std::endl;
   std::cout << "[INFO]: Solving with " << NFluxes - ( 1 + NomRegFluxLast - NomRegFluxFirst ) << " extra fluxes." << std::endl;
   // << std::endl;

   if (use_reg) {
     size_t NExtraFluxes = NFluxes - (1 + NomRegFluxLast - NomRegFluxFirst);
     std::cout << "NExtraFluxes : " << NExtraFluxes << std::endl;
     std::cout << "NFluxes : " << NFluxes << std::endl;


     if (!NExtraFluxes) {
       //std::cout << NExtraFluxes << " extra fluxes, using normal reg" << std::endl;
       for (size_t row_it = 0; row_it < (NFluxes - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
         FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
       }
       FluxMatrix_Solve(NEqs - 1, NFluxes - 1) = reg_param;
     }
     else {
       //std::cout << NExtraFluxes << " extra fluxes, using uncorrelated reg for extra fluxes" << std::endl;
       for (size_t row_it = 0; row_it < (NomRegFluxFirst - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param*BC_param;
       }
       //std::cout << "(NomRegFluxFirst - 1) " << (NomRegFluxFirst - 1) << std::endl;
       for (size_t row_it = (NomRegFluxFirst - 1); row_it < (NomRegFluxLast - 1)
					        && row_it < (NFluxes - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
         FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
       }
       //std::cout << "(NomRegFluxLast - 1) " << (NomRegFluxLast - 1) << std::endl;
       //std::cout << "(NFluxes - 1) " << (NFluxes - 1) << std::endl;
       //FluxMatrix_Solve( (NomRegFluxLast - 1) + NBins, (NomRegFluxLast - 1) ) = reg_param;
       for (size_t row_it = (NomRegFluxLast-1); row_it < NFluxes; ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param*BC_param;
       }
     }
/*
     else if (NExtraFluxes == NFluxes || NExtraFluxes >= NFluxes) {
       std::cout << "Assuming all " << NFluxes << " NFluxes are additional beam configs" << std::endl;
       for (size_t row_it = 0; row_it < NFluxes; ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param*BC_param;
       }
     }
*/

     if (ApplyWeightings && fParams.WFile.size()) {
       // FluxMatrix_Reduced = FluxMatrix_Solve;
       // FluxMatrix_Reduced.bottomRows(NFluxes) = FluxMatrix_Solve.bottomRows(NFluxes) * (FDWeights);
       // FluxMatrix_Solve = FluxMatrix_Reduced;
       //
      
       FluxMatrix_Solve.bottomRows(NFluxes) = (FluxMatrix_Solve.bottomRows(NFluxes) * FDWeights).eval();
       FluxMatrix_Reduced = FluxMatrix_Solve;
     }
     else {
       FluxMatrix_Reduced = FluxMatrix_Solve;
     }
   }
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FDWeights.inverse().format(CleanFmt) << std::endl;

    // std::cout << "FluxMatrix_Solve" << std::endl;
    // std::cout << "NFluxes " << NFluxes << std::endl;
    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FluxMatrix_Solve.format(CleanFmt) << std::endl;
    // std::cout << FluxMatrix_Solve.bottomRows(NFluxes).format(CleanFmt) << std::endl;

   while ( UseFluxesOld != UseFluxesNew ) {


    int EmptyFluxes = 0;
    for ( size_t check = 0; check < UseFluxesOld.size(); check++ ) {
      if ( !UseFluxesOld[check] ) {
	FluxMatrix_Reduced.col(check).topRows(NBins).setZero();
	EmptyFluxes++;
      }
    }
    std::cout << "\n ------    ------ " << std::endl;
    std::cout << " Empty Fluxes : " << EmptyFluxes << std::endl;

    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FluxMatrix_Reduced.format(CleanFmt) << std::endl;

    switch (fParams.algo_id) {
    case Params::kSVD: {
      if (use_reg) {
        last_solution =
            FluxMatrix_Reduced.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(Target);
      } else {
        last_solution = FluxMatrix_Reduced.topRows(NBins)
                            .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                            .solve(Target.topRows(NBins));
      }
      break;
    }
    case Params::kQR: {
      if (use_reg) {
        last_solution = FluxMatrix_Reduced.colPivHouseholderQr().solve(Target);
      } else {
        last_solution =
            FluxMatrix_Reduced.topRows(NBins).colPivHouseholderQr().solve(
                Target.topRows(NBins));
      }
      break;
    }
    case Params::kNormal: {
      if (use_reg) {
        last_solution = (FluxMatrix_Reduced.transpose() * FluxMatrix_Reduced)
                            .ldlt()
                            .solve(FluxMatrix_Reduced.transpose() * Target);
      } else {
        last_solution = (FluxMatrix_Reduced.transpose() * FluxMatrix_Reduced)
                            .topRows(NBins)
                            .ldlt()
                            .solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                                   Target.topRows(NBins));
      }
      break;
    }
    case Params::kInverse: {
      if (use_reg) {
        last_solution = ((FluxMatrix_Reduced.topRows(NBins).transpose() *
                          FluxMatrix_Reduced.topRows(NBins)) +
                         FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
                             FluxMatrix_Reduced.bottomRows(NFluxes))
                            .inverse() *
                        FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins);
      } else {
        last_solution = (FluxMatrix_Reduced.topRows(NBins).transpose() *
                         FluxMatrix_Reduced.topRows(NBins))
                            .inverse() *
                        FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins);
      }
      break;
    }
    case Params::kCOD: {
      if (use_reg) {
        last_solution = FluxMatrix_Reduced.completeOrthogonalDecomposition().solve(Target);
      } else {
        last_solution =
            FluxMatrix_Reduced.topRows(NBins).completeOrthogonalDecomposition().solve(
                Target.topRows(NBins));
      }
      break;
    }
    case Params::kConjugateGradient: {
      // SparseMatrix<double> = FluxMatrix_Reduced; // setting to sparsematrix - may not work - probably not optimal for runtimes
      // BiCGSTAB<SparseMatrix<double> > solver; // template
      //BiCGSTAB<Eigen::MatrixXd> solver; // 
      // ConjugateGradient<Eigen::MatrixXd>, Lower|Upper solver;
      // Eigen::ConjugateGradient<Eigen::MatrixXd, Lower|Upper> solver;
      
      Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> solver;
      std::cout << "ConjugateGradient solve" << std::endl;
      if (use_reg) {
	solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          		FluxMatrix_Reduced.topRows(NBins)) +
			FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          		FluxMatrix_Reduced.bottomRows(NFluxes));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
				     Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 

        /*((FluxMatrix_Reduced.topRows(NBins).transpose() *
          FluxMatrix_Reduced.topRows(NBins)) +
	  FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          FluxMatrix_Reduced.bottomRows(NFluxes)) * last_solution =
                        FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins);*/
      } else {
	solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
		 	FluxMatrix_Reduced.topRows(NBins));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 

	/*(FluxMatrix_Reduced.topRows(NBins).transpose() *
        (FluxMatrix_Reduced.topRows(NBins).transpose() *
	 FluxMatrix_Reduced.topRows(NBins)) * last_solution = 
                        FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins);*/
      }
      break;
    }
    case Params::kLeastSquaresConjugateGradient:  {
      Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd> solver;
      std::cout << "LeastSquaresConjugateGradient solve" << std::endl;
      if (use_reg) {
	solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          		FluxMatrix_Reduced.topRows(NBins)) +
			FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          		FluxMatrix_Reduced.bottomRows(NFluxes));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
				     Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      } else {
	solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
		 	FluxMatrix_Reduced.topRows(NBins));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      }
      break;
    }
    case Params::kBiCGSTAB:  {
      Eigen::BiCGSTAB<Eigen::MatrixXd> solver;
      std::cout << "BiCGSTAB solve" << std::endl;
      if (use_reg) {
	solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          		FluxMatrix_Reduced.topRows(NBins)) +
			FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          		FluxMatrix_Reduced.bottomRows(NFluxes));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
				     Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      } else {
	solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
		 	FluxMatrix_Reduced.topRows(NBins));

	last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                        Target.topRows(NBins));
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      }
      break;
    }
    case Params::kBiCGSTABguess:  {
      Eigen::BiCGSTAB<Eigen::MatrixXd> solver;
      std::cout << "BiCGSTAB solve with guess" << std::endl;
      if (use_reg) {
	solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          		FluxMatrix_Reduced.topRows(NBins)) +
			FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          		FluxMatrix_Reduced.bottomRows(NFluxes));

	if (soln_set) {
	  last_solution = solver.solveWithGuess(FluxMatrix_Reduced.topRows(NBins).transpose() *
						Target.topRows(NBins), last_solution).eval();
	}
	else {
	  last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
				       Target.topRows(NBins));
	}
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      } else {
	solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
		 	FluxMatrix_Reduced.topRows(NBins));

	if (soln_set) {
	  last_solution = solver.solveWithGuess(FluxMatrix_Reduced.topRows(NBins).transpose() *
                        			Target.topRows(NBins), last_solution).eval();
	}
	else {
	  last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                        	       Target.topRows(NBins));
	}
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl; 
      }
      soln_set = true;
      break;
    }
    }

    if (!last_solution.rows()) {
      res_norm = 0;
      soln_norm = 0;
      return last_solution;
    }

    res_norm = ((FluxMatrix_Reduced.topRows(NBins) * last_solution) -
                Target.topRows(NBins))
                   .squaredNorm();
    soln_norm = 0;
    if (reg_param > 0) {
      soln_norm =
          (FluxMatrix_Reduced.bottomRows(NFluxes) * last_solution / reg_param)
              .squaredNorm();
    }

    if ( isnan(res_norm) ) {
	std::cerr << "[ERROR] : NaN res norm found. " << std::endl;
	std::cerr << last_solution << std::endl;
	std::exit(EXIT_FAILURE);
    }

    if ( isnan(soln_norm) ) {
	std::cerr << "[ERROR] : NaN soln norm found. " << std::endl;
	std::cerr << last_solution << std::endl;
	std::exit(EXIT_FAILURE);
    }

    std::cout << "\n ------ LOOPING VALS ------ " << std::endl;
    std::cout << " res_norm = " << res_norm << std::endl;
    std::cout << " soln_norm = " << soln_norm << std::endl;

    UseFluxesNew = UseFluxesOld;
    /*for ( int sol_iter = 0; sol_iter < last_solution.rows(); sol_iter++) {
      // std::cout << " last_solution " << sol_iter << " : " << last_solution(sol_iter) << std::endl;
      if ( sol_iter < (NomRegFluxFirst - 1) || sol_iter > (NomRegFluxLast-1) ) {
        if ( std::abs(last_solution(sol_iter)) < fParams.coeffMagLower ) {
	  UseFluxesOld[sol_iter] = false; 
        }
      }
    }*/

    if (fParams.LeastNCoeffs) {
	std::cout << "fParams.LeastNCoeffs : " << fParams.LeastNCoeffs << std::endl; 
	UseFluxesOld = RemoveNCoeffs(last_solution, NFluxes);
    }
    else if (fParams.coeffMagLower) {
	std::cout << "fParams.coeffMagLower : " << fParams.coeffMagLower << std::endl; 
	UseFluxesOld = RemoveCoeffsSize(last_solution, NFluxes);
    }

    int NewEmptyFluxes = 0;
    for ( size_t check = 0; check < UseFluxesOld.size(); check++ ) {
      if ( !UseFluxesOld[check] ) {
	NewEmptyFluxes++;
      }
    }
    std::cout << "\n New Empty Fluxes : " << NewEmptyFluxes << std::endl;
    loops++;
   }
   std::cout << "\n ------ Loops : " << loops << " ------ " << std::endl;
   if (fParams.LeastNCoeffs && loops != 2) {
	std::cout << "[ERROR] : Loops != 2, implies RemoveNCoeffs not working correctly" << std::endl; 
        exit(1);
   }
   return last_solution;
  }
  Eigen::VectorXd SolveLast(double reg_param, double BC_param, double &res_norm, double &soln_norm) {
    // double dum1, dum2;
    std::cout << "reg_param = " << reg_param << std::endl; 
    std::cout << "BC_param = " << BC_param << std::endl; 
    RegParam = reg_param;
    BCParam = BC_param;
    CSNorm = 0;
    StabilityFactor = 0;
    // return Solve(reg_param, BC_param, dum1, dum2);
    return Solve(reg_param, BC_param, res_norm, soln_norm);
  }
  Eigen::VectorXd Solve(double reg_param = 0, double BC_param = 1) {
    double dum1, dum2;
    std::cout << "reg_param = " << reg_param << std::endl; 
    std::cout << "BC_param = " << BC_param << std::endl; 
    return Solve(reg_param, BC_param, dum1, dum2);
    
  }

  // std::vector<double> &CompressedSensingSolve(double reg_param, std::vector<double> omega, double &res_norm,
  Eigen::VectorXd const &CompressedSensingSolve(double reg_param, std::vector<double> &omega, double &res_norm,
                               double &soln_norm) {

   bool use_reg = reg_param > 0;
   size_t NFluxes = FluxMatrix_Solve.cols();
   size_t NEqs = FluxMatrix_Solve.rows();
   size_t NBins = NEqs - NFluxes;
   //double NomRegFluxFirst, NomRegFluxLast;
   UseFluxesOld.assign(NFluxes,true);
   UseFluxesNew.assign(NFluxes,false);
   int loops = 0;


   // std::cout << "[INFO]: Solving with " << NBins << " energy bins." << std::endl;
   std::cout << "[INFO]: Solving with " << NFluxes << " fluxes." << std::endl;
   // std::cout << "\n[INFO]: Solving with " << NumBeamConfigs << " extra configs in file." << std::endl;
   std::cout << "[INFO]: Solving with " << NFluxes - ( 1 + NomRegFluxLast - NomRegFluxFirst ) << " extra fluxes." << std::endl;
   // << std::endl;

   if (use_reg) {
     size_t NExtraFluxes = NFluxes - (1 + NomRegFluxLast - NomRegFluxFirst);
     if (NomRegFluxLast == 0 && NomRegFluxFirst == 0) { 
       NExtraFluxes = 0;
     }
     std::cout << "NExtraFluxes : " << NExtraFluxes << std::endl;
     std::cout << "NFluxes : " << NFluxes << std::endl;

     if (omega.size() == 0) {
       omega.assign(NFluxes, 1);
     }

     if (!NExtraFluxes) {
       for (size_t row_it = 0; row_it < (NFluxes - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
         FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
       }
       FluxMatrix_Solve(NEqs - 1, NFluxes - 1) = reg_param;
     }
     else {
       for (size_t row_it = 0; row_it < (NomRegFluxFirst - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = omega[row_it];
       }
       for (size_t row_it = (NomRegFluxFirst - 1); row_it < (NomRegFluxLast - 1)
					        && row_it < (NFluxes - 1); ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
         FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
       }
       //std::cout << "(NomRegFluxLast - 1) " << (NomRegFluxLast - 1) << std::endl;
       //std::cout << "(NFluxes - 1) " << (NFluxes - 1) << std::endl;
       //FluxMatrix_Solve( (NomRegFluxLast - 1) + NBins, (NomRegFluxLast - 1) ) = reg_param;
       for (size_t row_it = (NomRegFluxLast-1); row_it < NFluxes; ++row_it) {
         FluxMatrix_Solve(row_it + NBins, row_it) = omega[row_it];
       }
     }

     if (ApplyWeightings && fParams.WFile.size()) {
       FluxMatrix_Solve.bottomRows(NFluxes) = (FluxMatrix_Solve.bottomRows(NFluxes) * FDWeights).eval();
       FluxMatrix_Reduced = FluxMatrix_Solve;
     }
     else {
       FluxMatrix_Reduced = FluxMatrix_Solve;
     }
   }

    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FDWeights.inverse().format(CleanFmt) << std::endl;

    // std::cout << "FluxMatrix_Solve" << std::endl;
    // std::cout << "NFluxes " << NFluxes << std::endl;
    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << FluxMatrix_Solve.format(CleanFmt) << std::endl;
    // std::cout << FluxMatrix_Solve.bottomRows(NFluxes).format(CleanFmt) << std::endl;


   switch (fParams.algo_id) {
   case Params::kSVD: {
     if (use_reg) {
       last_solution =
           FluxMatrix_Reduced.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
               .solve(Target);
     } else {
       last_solution = FluxMatrix_Reduced.topRows(NBins)
                           .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                           .solve(Target.topRows(NBins));
     }
     break;
   }
   case Params::kQR: {
     if (use_reg) {
       last_solution = FluxMatrix_Reduced.colPivHouseholderQr().solve(Target);
     } else {
       last_solution =
           FluxMatrix_Reduced.topRows(NBins).colPivHouseholderQr().solve(
               Target.topRows(NBins));
     }
     break;
   }
   case Params::kNormal: {
     if (use_reg) {
       last_solution = (FluxMatrix_Reduced.transpose() * FluxMatrix_Reduced)
                           .ldlt()
                           .solve(FluxMatrix_Reduced.transpose() * Target);
     } else {
       last_solution = (FluxMatrix_Reduced.transpose() * FluxMatrix_Reduced)
                           .topRows(NBins)
                           .ldlt()
                           .solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                                  Target.topRows(NBins));
     }
     break;
   }
   case Params::kInverse: {
     if (use_reg) {
       last_solution = ((FluxMatrix_Reduced.topRows(NBins).transpose() *
                         FluxMatrix_Reduced.topRows(NBins)) +
                        FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
                            FluxMatrix_Reduced.bottomRows(NFluxes))
                           .inverse() *
                       FluxMatrix_Reduced.topRows(NBins).transpose() *
                       Target.topRows(NBins);
     } else {
       last_solution = (FluxMatrix_Reduced.topRows(NBins).transpose() *
                        FluxMatrix_Reduced.topRows(NBins))
                           .inverse() *
                       FluxMatrix_Reduced.topRows(NBins).transpose() *
                       Target.topRows(NBins);
     }
     break;
   }
   case Params::kCOD: {
     if (use_reg) {
       last_solution = FluxMatrix_Reduced.completeOrthogonalDecomposition().solve(Target);
     } else {
       last_solution =
           FluxMatrix_Reduced.topRows(NBins).completeOrthogonalDecomposition().solve(
               Target.topRows(NBins));
     }
     break;
   }
   case Params::kConjugateGradient: {
     Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> solver;
     std::cout << "ConjugateGradient solve" << std::endl;
     if (use_reg) {
       solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
         	       FluxMatrix_Reduced.topRows(NBins)) +
		       FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          	       FluxMatrix_Reduced.bottomRows(NFluxes));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
			            Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     } else {
       solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
	              FluxMatrix_Reduced.topRows(NBins));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                       Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     }
     break;
   }
   case Params::kLeastSquaresConjugateGradient:  {
     Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd> solver;
     std::cout << "LeastSquaresConjugateGradient solve" << std::endl;
     if (use_reg) {
       solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
         	       FluxMatrix_Reduced.topRows(NBins)) +
		       FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
           	       FluxMatrix_Reduced.bottomRows(NFluxes));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
  			            Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     } else {
       solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
	   	      FluxMatrix_Reduced.topRows(NBins));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                       Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     }
     break;
   }
   case Params::kBiCGSTAB:  {
     Eigen::BiCGSTAB<Eigen::MatrixXd> solver;
     std::cout << "BiCGSTAB solve" << std::endl;
     if (use_reg) {
       solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          	       FluxMatrix_Reduced.topRows(NBins)) +
		       FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          	       FluxMatrix_Reduced.bottomRows(NFluxes));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
				    Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     } else {
       solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
      	 	      FluxMatrix_Reduced.topRows(NBins));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                       Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     }
     break;
   }
   case Params::kBiCGSTABguess:  {
     Eigen::BiCGSTAB<Eigen::MatrixXd> solver;
     std::cout << "BiCGSTAB solve with guess" << std::endl;
     if (use_reg) {
       solver.compute((FluxMatrix_Reduced.topRows(NBins).transpose() *
          	       FluxMatrix_Reduced.topRows(NBins)) +
		       FluxMatrix_Reduced.bottomRows(NFluxes).transpose() *
          	       FluxMatrix_Reduced.bottomRows(NFluxes));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
			       Target.topRows(NBins));
       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     } else {
       solver.compute(FluxMatrix_Reduced.topRows(NBins).transpose() *
 	 	      FluxMatrix_Reduced.topRows(NBins));

       last_solution = solver.solve(FluxMatrix_Reduced.topRows(NBins).transpose() *
                       	            Target.topRows(NBins));

       std::cout << "#iterations:     " << solver.iterations() << std::endl;
       std::cout << "estimated error: " << solver.error()      << std::endl; 
     }
     break;
   }
   }

   if (!last_solution.rows()) {
     res_norm = 0;
     soln_norm = 0;
     // return last_solution;
     std::vector<double> fail{0};
     // return fail; 
   }

   res_norm = ((FluxMatrix_Reduced.topRows(NBins) * last_solution) -
               Target.topRows(NBins))
                  .squaredNorm();
   soln_norm = 0;
   if (reg_param > 0) {
     soln_norm =
         (FluxMatrix_Reduced.bottomRows(NFluxes) * last_solution / reg_param)
             .squaredNorm();
   }

   if ( isnan(res_norm) ) {
	std::cerr << "[ERROR] : NaN res norm found. " << std::endl;
	std::cerr << last_solution << std::endl;
	std::exit(EXIT_FAILURE);
   }

   if ( isnan(soln_norm) ) {
	std::cerr << "[ERROR] : NaN soln norm found. " << std::endl;
	std::cerr << last_solution << std::endl;
	std::exit(EXIT_FAILURE);
   }

   std::cout << " res_norm = " << res_norm << std::endl;
   std::cout << " soln_norm = " << soln_norm << std::endl;

   std::vector<double> coeffvec(last_solution.data(), last_solution.data() +
 					last_solution.rows() * last_solution.cols());
   omega = coeffvec;
   return last_solution;
   // return coeffvec;
  }

  Eigen::VectorXd CompressedSensingSolveLast(double reg_param, std::vector<double> &omega, double &res_norm, double &soln_norm, double csnorm, double stabfactor) {
    // double dum1, dum2;
    std::cout << "reg_param = " << reg_param << std::endl; 
    std::cout << "BC_param = " << 0 << std::endl; 
    std::cout << "CSNorm = " << csnorm << std::endl; 
    std::cout << "Stability Factor = " << stabfactor << std::endl; 

    RegParam = reg_param;
    BCParam = 0;
    CSNorm = csnorm;
    StabilityFactor = stabfactor;
    // return Solve(reg_param, BC_param, dum1, dum2);
    return CompressedSensingSolve(reg_param, omega, res_norm, soln_norm);
  }


  std::vector<bool> RemoveNCoeffs(Eigen::VectorXd last_solution, size_t NFluxes) {
    std::vector<double> coeffVec(last_solution.data(), last_solution.data() + last_solution.rows() * last_solution.cols());
   /* 
    std::cout << "coeffVec.size() : " << coeffVec.size() << std::endl;
    std::cout << "NomRegFluxFirst :" << NomRegFluxFirst << std::endl;
    std::cout << "NomRegFluxLast :" << NomRegFluxLast << std::endl;
    std::cout << "coeffVec elements : " << std::endl;
    int counter = 0;
    */
    for (double &cV_it : coeffVec) {
	cV_it = std::abs(cV_it);	
	// std::cout << "coeffVec[" << counter << "]" << cV_it << std::endl;
	// counter++;
    }
    
    std::vector<bool> newFluxVec(NFluxes,true);
    // int ParN = fParams.LeastNCoeffs;
    // int ParN = NFluxes - fParams.LeastNCoeffs;
    size_t NExtraFluxes = NFluxes - (1 + NomRegFluxLast - NomRegFluxFirst);
    int ParN = NExtraFluxes - fParams.LeastNCoeffs;
    for (int elem = 0; elem < ParN; elem++) {
	// std::cout << "k : " << elem << std::endl;
	// std::cout << "ParN : " << ParN << std::endl;
	int k_index = min_k(coeffVec.begin(), coeffVec.end(), elem);
	if ( k_index >= (NomRegFluxFirst-1) && k_index <= (NomRegFluxLast-1) ) {
	    ParN++;
	    continue;
	}
	// std::cout << "(NomRegFluxFirst-1) : " << (NomRegFluxFirst-1) << std::endl;
	// std::cout << "(NomRegFluxLast-1) : " << (NomRegFluxLast-1) << std::endl;
	// std::cout << "k : " << elem << std::endl;
	// std::cout << k_index << std::endl;
	// std::cout << "coeffVec[" << k_index << "] = " << coeffVec[k_index] << "\n" << std::endl;
	// newFluxVec[k_index] = false; 
	if (newFluxVec[k_index]) {
	    newFluxVec[k_index] = false; 
	}
	else {
	    std::cout << "[ERROR] newFluxVec[" << k_index << "] : already scanned. Double-counting flux no.: " << elem << std::endl;
	    exit(1);
	}
    }
    return newFluxVec;
  }

  std::vector<bool> RemoveCoeffsSize(Eigen::VectorXd last_solution, size_t NFluxes) {
    // std::vector<bool> newFluxVec(NFluxes,true);
    std::vector<bool> newFluxVec = UseFluxesOld;
    // std::cout << "NomRegFluxFirst :" << NomRegFluxFirst << std::endl;
    // std::cout << "NomRegFluxLast :" << NomRegFluxLast << std::endl;
    for ( int sol_iter = 0; sol_iter < last_solution.rows(); sol_iter++) {
      // std::cout << " last_solution " << sol_iter << " : " << last_solution(sol_iter) << std::endl;
      if ( sol_iter < (NomRegFluxFirst - 1) || sol_iter > (NomRegFluxLast-1) ) {
        if ( std::abs(last_solution(sol_iter)) < fParams.coeffMagLower ) {
	  newFluxVec[sol_iter] = false; 
        }
      }
    }
    return newFluxVec;
  }

  void BuildWeights(std::vector<std::string> HistNames) {

   // GetFlux
   std::vector<Eigen::MatrixXd> FluxWeightMatrices;
   double nomFluxWeight = 0;
   for (size_t hist_it=0; hist_it < HistNames.size(); hist_it++) {
     if (fParams.WFile.size() && HistNames.size()) {
       ApplyWeightings = true;

       std::unique_ptr<TH2> inpFluxWeightHist =
           GetHistogram<TH2>(fParams.WFile, HistNames[hist_it]);

       if (!inpFluxWeightHist) {
         std::cout << "[ERROR]: Found no input FD weight flux with name: \"" 
                   << fParams.WFile << "\" in file: \"" 
                   << HistNames[hist_it] << "\"." << std::endl;
         throw;
       }
       std::unique_ptr<TH2> FluxWeightHist(static_cast<TH2 *>(inpFluxWeightHist->Clone()));
       FluxWeightHist->SetDirectory(nullptr);
////////////////////////////////////////////
   // OscillateFlux here ? Maybe hijack other function
///////////////////////////////////////////
//

/*
       int lowCurrBin;
       if ( fParams.CurrentRangeSplit.size() ) {
         lowCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.CurrentRangeSplit[hist_it].first );
       } else {
         lowCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.CurrentRange.first );
       }
       if ( lowCurrBin == 0 ) {
	 lowCurrBin = 1; // If below bin range, set to minimum non-underflow bin
       }
       int nomCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.NominalFlux.second );

       int highCurrBin;
       if ( fParams.CurrentRangeSplit.size() ) {
         highCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.CurrentRangeSplit[hist_it].second );
       } else {
         highCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.CurrentRange.second );
       }
       if ( highCurrBin == (FluxWeightHist->GetYaxis()->GetNbins() + 1) ) {
 	 highCurrBin -= 1; // If above bin range, set to maximum non-overflow bin
       }
       int yBins = ( highCurrBin + 1 ) - lowCurrBin;
*/

       int nomCurrBin = FluxWeightHist->GetYaxis()->FindFixBin( fParams.NominalFlux.second );
       int yBins = AllZbins[hist_it].size();

       Eigen::MatrixXd InitWeightMat = Eigen::MatrixXd::Zero(FluxWeightHist->GetXaxis()->GetNbins(), yBins);
       for (Int_t ybi_it = 0; ybi_it < yBins; 
            ++ybi_it) {
         for (Int_t ebi_it = 0; ebi_it < FluxWeightHist->GetXaxis()->GetNbins();
              ++ebi_it) {
           InitWeightMat(ebi_it, ybi_it) =
               FluxWeightHist->GetBinContent(ebi_it + 1, AllZbins[hist_it][ybi_it]);
         }
       }

   // Transform InitWeightMat 
       Int_t InitMrows = InitWeightMat.rows();
       Int_t InitMcols = InitWeightMat.cols();

       switch (fParams.WeightMethod) {
       case Params::TotalFlux: {
         for (Int_t ybi_it = 0; ybi_it < InitMcols;
              ++ybi_it) {
           double ColumnSum = 0;
           for (Int_t ebi_it = 0; ebi_it < InitMrows;
                ++ebi_it) {
             ColumnSum += InitWeightMat(ebi_it, ybi_it);
  	     // InitWeightMat(ebi_it, ybi_it) = 0;
           }
	   InitWeightMat.col(ybi_it).setZero();
	   InitWeightMat(ybi_it, ybi_it) = 1.0/ColumnSum;
         }
         break;
       }
       case Params::MaxFlux: {
         for (Int_t ybi_it = 0; ybi_it < InitMcols;
              ++ybi_it) {
           double ColumnMax = 0;
           for (Int_t ebi_it = 0; ebi_it < InitMrows;
                ++ebi_it) {
	     if (InitWeightMat(ebi_it, ybi_it) > ColumnMax) {
               ColumnMax = InitWeightMat(ebi_it, ybi_it);
             }
  	     // InitWeightMat(ebi_it, ybi_it) = 0;
           }
	   InitWeightMat.col(ybi_it).setZero();
           InitWeightMat(ybi_it, ybi_it) = 1.0/ColumnMax;
         }
         break;
       }
       }


   // Scale up InitWeightMat to match number of OA flux slices
       Int_t FullMcols = 0;
       for (size_t v2 = 0; v2 < AllOAbins[hist_it].size(); v2++) {
         FullMcols += AllOAbins[hist_it][v2];
       }

       Eigen::MatrixXd TmpWeightMat = Eigen::MatrixXd::Zero(FullMcols, FullMcols);
       for (Int_t ybi_it = 0; ybi_it < yBins; ++ybi_it) {
         for (Int_t ebi_it = 0; ebi_it < yBins; ++ebi_it) {
	   int FluxesPerZ = AllOAbins[hist_it][ybi_it];
	   // std::cout << "FluxesPerZ : " << FluxesPerZ << std::endl;
           for (Int_t fpz_it = 0; fpz_it < FluxesPerZ; fpz_it++) {
             TmpWeightMat(ebi_it*FluxesPerZ + fpz_it, ybi_it*FluxesPerZ + fpz_it) = InitWeightMat(ebi_it, ybi_it);
	   }
	 }
       }

       if (hist_it == (fParams.NominalFlux.first - 1) ) {
	 // Get iterator to nominal z bin
    	 std::vector<size_t>::iterator it = std::find( AllZbins[hist_it].begin(), AllZbins[hist_it].end(), nomCurrBin );
	 if (it == AllZbins[hist_it].end()) {
	   std::cout << "[ERROR] : Nominal current bin not in current range from FD weights" << std::endl;
           exit(1);
	 }
	 // Get nom index as distance from start to iterator
	 int index = std::distance(AllZbins[hist_it].begin(), it);

	 int prev_fluxes = 0;
	 for ( int prev_it = 0; prev_it < index; prev_it++ ) {
	   prev_fluxes += AllOAbins[hist_it][prev_it];
	 }
	 // std::cout << "prev_fluxes : " << prev_fluxes << std::endl;
         nomFluxWeight = TmpWeightMat( prev_fluxes, prev_fluxes );
	 std::cout << "nomFluxWeight : " << nomFluxWeight << std::endl;
       }
       FluxWeightMatrices.emplace_back(TmpWeightMat);
     }
   }

   Int_t FDWtotrows = 0;
   for (size_t v1 = 0; v1 < AllOAbins.size(); v1++) {
     for (size_t v2 = 0; v2 < AllOAbins[v1].size(); v2++) {
       FDWtotrows += AllOAbins[v1][v2];
     }
   }

   /* std::vector<Int_t> OAbinsperhist(AllOAbins.size(), 0);
   OAbinsperhist.assign(AllOAbins.size(), 0);
   for (size_t v1 = 0; v1 < AllOAbins.size(); v1++) {
     for (size_t v2 = 0; v2 < AllOAbins[v1].size(); v2++) {
       OAbinsperhist[v1] += AllOAbins[v1][v2];
     }
   }*/

   // StoreFlux
   FDWeights = Eigen::MatrixXd::Zero(FDWtotrows, FDWtotrows);
   Int_t FDWsubrows = 0;
   for (size_t n = 0; n < FluxWeightMatrices.size(); n++) {
     // std::cout << " FDWsubrows :" << FDWsubrows << std::endl;
     FDWeights.block(0 + FDWsubrows, 0 + FDWsubrows, OAbinsperhist[n], OAbinsperhist[n]) = FluxWeightMatrices[n];
     FDWsubrows += OAbinsperhist[n];
   }
   FDWeights /= nomFluxWeight;
   // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
   // std::cout << FDWeights.format(CleanFmt) << std::endl;
  }


  // void Store(TDirectory *td, double res_norm = 0, double soln_norm = 0) {

  void Write(TDirectory *td, double res_norm = 0, double soln_norm = 0) {
    if (!last_solution.rows()) {
      return;
    }

    size_t NFluxes = FluxMatrix_Solve.cols();
    size_t NEqs = FluxMatrix_Solve.rows();
    size_t NBins = NEqs - NFluxes;

    TH1 *FDFlux_osc_wr = static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_osc"));
    TH1 *FDFlux_targ_OORScale =
        static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_targ_OORScale"));
    FDFlux_targ_OORScale->Reset();
    FillHistFromEigenVector(FDFlux_targ_OORScale, Target, low_offset);

    TH1 *FDFlux_targ = static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_targ"));
    FDFlux_targ->Reset();
    Eigen::VectorXd Target_rescale = Target;
    Target_rescale.topRows(FitIdxLow).array() /= fParams.OORFactor;
    // Target_rescale.bottomRows(NBins - FitIdxHigh).array() /= fParams.OORFactor;
    Target_rescale.bottomRows(NEqs - (FitIdxHigh + 1) ).array() /= fParams.OORFactor;
    FillHistFromEigenVector(FDFlux_targ, Target_rescale, low_offset);

    TH1 *FDFlux_bf = static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_bf"));
    FDFlux_bf->Reset();

    Eigen::VectorXd bf = (FluxMatrix_Full * last_solution);
    FillHistFromEigenVector(FDFlux_bf, bf, low_offset);

    TH1D *Coeffs =
        new TH1D("Coeffs", "", last_solution.rows(), 0, last_solution.rows());
    FillHistFromEigenVector(Coeffs, last_solution);
    Int_t xval = 0; 
    for (int bi_it = 0; bi_it < OAbinsperhist.size(); bi_it++) { 
      Int_t xval2 = xval;
      for (int bi2_it = 0; bi2_it < AllOAbins[bi_it].size(); bi2_it++) {
	xval2 += AllOAbins[bi_it][bi2_it];
        TLine *l = new TLine(xval2,Coeffs->GetMinimum(),xval2,Coeffs->GetMaximum());
	l->SetLineColor(kGreen);
        Coeffs->GetListOfFunctions()->Add(l);
      }
      xval += OAbinsperhist[bi_it];
      TLine *l = new TLine(xval,Coeffs->GetMinimum(),xval,Coeffs->GetMaximum());
      l->SetLineColor(kRed);
      // TLine *l = new TLine(20,-1,20,1);
      Coeffs->GetListOfFunctions()->Add(l);
    }

    TH1D *RegDiag=
        new TH1D("RegMatrixDiagonal", "", NFluxes, 0, NFluxes);
    Eigen::VectorXd RegDiagVec = last_solution;
    RegDiagVec.setZero();
    for (int reg_iter = 0; reg_iter < NFluxes; reg_iter++) {
        RegDiagVec(reg_iter) = FluxMatrix_Reduced.bottomRows(NFluxes)(reg_iter, reg_iter);

	// std::cout << "[" << reg_iter << "," << reg_iter << "] = " << FluxMatrix_Reduced.bottomRows(NFluxes)(reg_iter, reg_iter) << std::endl;
	// if (reg_iter!=NFluxes-1) {
	    // std::cout << "[" << reg_iter << "," << reg_iter+1 << "] = " << FluxMatrix_Reduced.bottomRows(NFluxes)(reg_iter, reg_iter+1) << std::endl;
	// }
    }

    /*TH1D *RegDiag=
        new TH1D("RegMatrixDiagonal", "", NFluxes, 0, NFluxes);
    Eigen::VectorXd RegDiagVec = last_solution.setZero();
    for (int reg_iter = 0; reg_iter < 1+(NFluxes)/2; reg_iter+=2) {
        RegDiagVec(reg_iter) = FluxMatrix_Reduced.bottomRows(NFluxes)(reg_iter, reg_iter);
        RegDiagVec(reg_iter+1) = FluxMatrix_Reduced.bottomRows(NFluxes)(reg_iter, reg_iter+1);
    }*/
    FillHistFromEigenVector(RegDiag, RegDiagVec);
    /*Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << FluxMatrix_Reduced.format(CleanFmt) << std::endl;
    std::cout << "FluxMatrix_Reduced.rows() : " << FluxMatrix_Reduced.rows() << std::endl;
    std::cout << "FluxMatrix_Reduced.cols() : " << FluxMatrix_Reduced.cols() << std::endl;

    std::cout << "FDWeights :" << std::endl;
    std::cout << FDWeights.format(CleanFmt) << std::endl;

    std::cout << "last_solution:" << std::endl;
    std::cout << last_solution.format(CleanFmt) << std::endl;*/

    TTree *ParTree =
        new TTree("Params", "Params"); 
    ParTree->Branch("RegParam", &RegParam, "RegParam/D");
    ParTree->Branch("BCParam", &BCParam, "BCParam/D");
    ParTree->Branch("res_norm", &res_norm, "res_norm/D");
    ParTree->Branch("soln_norm", &soln_norm, "soln_norm/D");
    ParTree->Branch("CSNorm", &CSNorm, "CSNorm/D");
    ParTree->Branch("StabilityFactor", &StabilityFactor, "StabilityFactor/D");
    ParTree->Fill();

    if (BCTree) { 
      static_cast<TTree *>(BCTree->Clone("ConfigTree"))->SetDirectory(td);
    }

    if (FDFlux_unosc) {
      static_cast<TH1 *>(FDFlux_unosc->Clone("FDFlux_unosc"))->SetDirectory(td);
    }

    FDFlux_osc_wr->SetDirectory(td);
    FDFlux_targ->SetDirectory(td);
    FDFlux_targ_OORScale->SetDirectory(td);
    FDFlux_bf->SetDirectory(td);
    Coeffs->SetDirectory(td);
    ParTree->SetDirectory(td);
    // ParTree->Write();
//    BCTree->SetDirectory(td);
  }

  void WriteCS(TDirectory *td, double res_norm = 0, double soln_norm = 0) {
    if (!last_solution.rows()) {
      return;
    }

    size_t NFluxes = FluxMatrix_Solve.cols();
    size_t NEqs = FluxMatrix_Solve.rows();
    size_t NBins = NEqs - NFluxes;

    TH1 *FDFlux_prefilter = static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_prefilter"));
    FDFlux_prefilter->Reset();

    Eigen::VectorXd bf = (FluxMatrix_Full * last_solution);
    FillHistFromEigenVector(FDFlux_prefilter, bf, low_offset);

    TH1D *Coeffs_prefilter =
        new TH1D("Coeffs_prefilter", "", last_solution.rows(), 0, last_solution.rows());
    FillHistFromEigenVector(Coeffs_prefilter, last_solution);
    Int_t xval = 0; 
    for (int bi_it = 0; bi_it < OAbinsperhist.size(); bi_it++) { 
      Int_t xval2 = xval;
      for (int bi2_it = 0; bi2_it < AllOAbins[bi_it].size(); bi2_it++) {
	xval2 += AllOAbins[bi_it][bi2_it];
        TLine *l = new TLine(xval2,Coeffs_prefilter->GetMinimum(),xval2,Coeffs_prefilter->GetMaximum());
	l->SetLineColor(kGreen);
        Coeffs_prefilter->GetListOfFunctions()->Add(l);
      }
      xval += OAbinsperhist[bi_it];
      TLine *l = new TLine(xval,Coeffs_prefilter->GetMinimum(),xval,Coeffs_prefilter->GetMaximum());
      l->SetLineColor(kRed);
      // TLine *l = new TLine(20,-1,20,1);
      Coeffs_prefilter->GetListOfFunctions()->Add(l);
    }

    TTree *ParTree =
        new TTree("Params", "Params"); 
    ParTree->Branch("res_norm_pref", &res_norm, "res_norm_pref/D");
    ParTree->Branch("soln_norm_pref", &soln_norm, "soln_norm_pref/D");
    ParTree->Fill();

    FDFlux_prefilter->SetDirectory(td);
    Coeffs_prefilter->SetDirectory(td);
    ParTree->SetDirectory(td);
    // ParTree->Write();
//    BCTree->SetDirectory(td);
  }
};

#ifdef FLS_WRAP_IN_NAMESPACE
}
#endif

#endif
