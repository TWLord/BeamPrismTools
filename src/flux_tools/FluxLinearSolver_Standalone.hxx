#ifndef FLUXLINEARSOLVER_STANDALONE_HXX_SEEN
#define FLUXLINEARSOLVER_STANDALONE_HXX_SEEN

#include "BargerPropagator.h"

#include "Eigen/Dense"
#include "Eigen/StdVector"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    enum Solver { kSVD = 1, kQR, kNormal, kInverse };

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

    bool FitBetweenFoundPeaks;
    std::pair<double, double> FitBetween;

    /// Number of ENu bins to merge
    int MergeENuBins;
    /// Number of OA position bins to merge
    int MergeOAPBins;

    /// Use a subset of the full input ranges described by this descriptor
    ///
    std::string OffAxisRangesDescriptor;

    std::pair<double, double> CurrentRange;

    std::string HConfigs;
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
    p.ExpDecayRate = 3;
    p.CurrentRange = {0, 350};
    p.HConfigs = "1,2,3,4,5";

    return p;
  }

  Eigen::MatrixXd FluxMatrix_Full;
  Eigen::MatrixXd FluxMatrix_zSlice;
  Eigen::MatrixXd FluxMatrix_All; 
  Eigen::MatrixXd FluxMatrix_Solve;
  Eigen::VectorXd Target;

  Eigen::VectorXd last_solution;

  std::unique_ptr<TH1> FDFlux_unosc;
  std::unique_ptr<TH1> FDFlux_osc;

  std::unique_ptr<TTree> BCTree;

  Int_t NumBeamConfigs;
  size_t NCoefficients;
  std::vector<size_t> nZbins;
  size_t FluxesPerZ;
  size_t low_offset, FitIdxLow, FitIdxHigh;

  void
  Initialize(Params const &p,
             std::pair<std::string, std::string> NDFluxDescriptor = {"", ""},
             std::pair<std::string, std::string> FDFluxDescriptor = {"", ""},
             std::pair<std::string, std::string> NDBeamConfDescriptor = {"", ""},
             bool FDIsOsc = false) {

    std::pair<std::string, std::vector<std::string>> vectNDFluxDescriptor;
    vectNDFluxDescriptor.first = NDFluxDescriptor.first;
    vectNDFluxDescriptor.second.emplace_back(NDFluxDescriptor.second);

    Initialize(p, vectNDFluxDescriptor, FDFluxDescriptor,
		NDBeamConfDescriptor, FDIsOsc);

  }

  void
  Initialize(Params const &p,
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
//////////////////////////////// editing above ^

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
    TFile *BCfile = CheckOpenFile(NDFluxDescriptor.first);
    TTree *BCtree = (TTree*)BCfile->Get((NDBeamConfDescriptor.first).c_str());
    if (!BCtree) {
      std::cout << "Couldn't find config tree: " << (NDBeamConfDescriptor.first).c_str() << std::endl;
      NumBeamConfigs=0;
      std::cout << "Number of additional Beam Configs used = " << NumBeamConfigs << std::endl;
      return;
    }
    BCtree = (TTree*)BCfile->Get((NDBeamConfDescriptor.first).c_str());
    BCtree->SetBranchAddress("NumBeamConfigs",&NumBeamConfigs); 
    BCtree->GetEntry(0);
    std::cout << "Number of additional Beam Configs used = " << NumBeamConfigs << std::endl;
  }

  void SetNDFluxes(std::vector<std::unique_ptr<TH3>> const &NDFluxes, bool ApplyXRanges = true) {

    std::vector<std::pair<double, double>> XRanges;
    if (ApplyXRanges && fParams.OffAxisRangesDescriptor.size()) {
      XRanges = BuildRangesList(fParams.OffAxisRangesDescriptor);
    }

    std::vector<int> Confs;
    if (fParams.HConfigs.size()) {
      Confs = BuildHConfsList(fParams.HConfigs);
    }

    std::cout << NDFluxes[0]->GetNbinsZ() << std::endl;
    //std::vector<Eigen::MatrixXd> NDMatrices(NDFluxes.size()*NDFluxes[0]->GetNbinsZ());
    std::vector<Eigen::MatrixXd> NDMatrices;
    std::cout << " For initialising matrix vec, NbinsZ : " << (NDFluxes.size()*NDFluxes[0]->GetNbinsZ()) << std::endl;
    std::cout << " matrix vec size : " << NDMatrices.size() << std::endl;

    for (size_t i = 0; i < NDFluxes.size(); i++ ) {
      std::unique_ptr<TH3> Flux3D(static_cast<TH3 *>(NDFluxes[i]->Clone()));
      Flux3D->SetDirectory(nullptr);

      // might want FindFixBin?
      // lowCurrentBin = Flux3D->GetZaxis()->FindBin( CurrentRange.first );
      // highCurrentBin = Flux3D->GetZaxis()->FindBin( CurrentRange.second );

      int lowCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRange.first );
      int highCurrentBin = Flux3D->GetZaxis()->FindFixBin( fParams.CurrentRange.second );
      int zBins = ( highCurrentBin + 1 ) - lowCurrentBin;
      nZbins.emplace_back(zBins);
      std::cout << " lowCurrentBin : " << lowCurrentBin << std::endl;
      std::cout << " highCurrentBin : " << highCurrentBin << std::endl; 
      std::cout << " zBins : " << zBins << std::endl; 


      for (int z = lowCurrentBin; z <= highCurrentBin; z++) {
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

        if (XRanges.size()) { // Use a subset of (possibly merged) off-axis slices
          std::vector<std::unique_ptr<TH1>> FluxSlices =
              MergeSplitTH2(Flux2D, true, XRanges);
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
//	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
//	std::cout << FluxMatrix_zSlice.format(CleanFmt) << std::endl;
    	FluxesPerZ = FluxMatrix_zSlice.cols();
    
      }
    }
    Int_t NDrows = NDMatrices[0].rows();
    Int_t NDcols = NDMatrices[0].cols();

    std::cout << "NDrows : " << NDrows << std::endl;
    std::cout << "NDcols : " << NDcols << std::endl;

    std::cout << "NDrows : " << NDMatrices[1].rows() << std::endl;
    std::cout << "NDcols : " << NDMatrices[1].cols() << std::endl;

    std::cout << "NDMatrices.size : " << NDMatrices.size() << std::endl;

    FluxMatrix_Full = Eigen::MatrixXd(NDrows, NDcols*NDMatrices.size());
    for (size_t n = 0; n < NDMatrices.size(); n++) {
//      for (Int_t NDrow_it = 0; NDrow_it < NDrows) { 
//        for (Int_t NDcol_it = 0; NDcol_it < NDcols) { 
//	  FluxMatrix_Full( NDrow_it + n*NDrows, NDcol_it )
//        }
//      }
      std::cout << "n*NDrows : " << n*NDrows << std::endl;
      std::cout << "(n+1)*NDrows : " << (n+1)*NDrows << std::endl;
      std::cout << "NDcols : " << NDcols << std::endl;
      // FluxMatrix_Full.block(0 + n*NDrows, 0, NDrows, NDcols) = NDMatrices[n];
      FluxMatrix_Full.block(0, 0 + n*NDcols, NDrows, NDcols) = NDMatrices[n];
    }
    //Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    //std::cout << FluxMatrix_Full.format(CleanFmt) << std::endl;
    NCoefficients = FluxMatrix_Full.cols();

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

    std::cout << "[INFO]: Solving with " << NBins << " energy bins." << std::endl;
    std::cout << "[INFO]: Solving with " << NFluxes << " fluxes." << std::endl;
    // std::cout << "[INFO]: Solving with " << NumBeamConfigs << " extra fluxes." << std::endl;
    std::cout << "[INFO]: Solving with " << (nZbins[0] - 1)*FluxesPerZ << " extra fluxes." << std::endl;
    // << std::endl;

    if (use_reg) {
      //size_t NExtraFluxes = NumBeamConfigs;
      size_t NExtraFluxes = (nZbins[0] - 1)*FluxesPerZ;
      std::cout << "NExtraFluxes : " << NExtraFluxes << std::endl;
      std::cout << "FluxesPerZ : " << FluxesPerZ << std::endl;
      for (size_t z=1; z < nZbins.size(); z++) {
        NExtraFluxes += (nZbins[z]*FluxesPerZ);
      }
      std::cout << "NExtraFluxes : " << NExtraFluxes << std::endl;

      if (!NExtraFluxes) {
	std::cout << NExtraFluxes << " extra fluxes, using normal reg" << std::endl;
        for (size_t row_it = 0; row_it < (NFluxes - 1); ++row_it) {
          FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
          FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
        }
        FluxMatrix_Solve(NEqs - 1, NFluxes - 1) = reg_param;
      }
      else if (NExtraFluxes == NFluxes || NExtraFluxes >= NFluxes) {
	std::cout << "Assuming all " << NFluxes << " NFluxes are additional beam configs" << std::endl;
        for (size_t row_it = 0; row_it < NFluxes; ++row_it) {
          FluxMatrix_Solve(row_it + NBins, row_it) = reg_param*BC_param;
        }
      }
      else {
	std::cout << NExtraFluxes << " extra fluxes, using uncorrelated reg for extra fluxes" << std::endl;
        for (size_t row_it = 0; row_it < (NFluxes - (1+NExtraFluxes)); ++row_it) {
          FluxMatrix_Solve(row_it + NBins, row_it) = reg_param;
          FluxMatrix_Solve(row_it + NBins, row_it + 1) = -reg_param;
        }
	FluxMatrix_Solve((NFluxes - (1+NExtraFluxes)) + NBins, (NFluxes - (1+NExtraFluxes))) = reg_param;
        for (size_t row_it = (NFluxes - NExtraFluxes); row_it < NFluxes; ++row_it) {
          FluxMatrix_Solve(row_it + NBins, row_it) = reg_param*BC_param;
        }
      }
    }
//    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
//    std::cout << FluxMatrix_Solve.format(CleanFmt) << std::endl;

    switch (fParams.algo_id) {
    case Params::kSVD: {
      if (use_reg) {
        last_solution =
            FluxMatrix_Solve.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(Target);
      } else {
        last_solution = FluxMatrix_Solve.topRows(NBins)
                            .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                            .solve(Target.topRows(NBins));
      }
      break;
    }
    case Params::kQR: {
      if (use_reg) {
        last_solution = FluxMatrix_Solve.colPivHouseholderQr().solve(Target);
      } else {
        last_solution =
            FluxMatrix_Solve.topRows(NBins).colPivHouseholderQr().solve(
                Target.topRows(NBins));
      }
      break;
    }
    case Params::kNormal: {
      if (use_reg) {
        last_solution = (FluxMatrix_Solve.transpose() * FluxMatrix_Solve)
                            .ldlt()
                            .solve(FluxMatrix_Solve.transpose() * Target);
      } else {
        last_solution = (FluxMatrix_Solve.transpose() * FluxMatrix_Solve)
                            .topRows(NBins)
                            .ldlt()
                            .solve(FluxMatrix_Solve.topRows(NBins).transpose() *
                                   Target.topRows(NBins));
      }
      break;
    }
    case Params::kInverse: {
      if (use_reg) {
        last_solution = ((FluxMatrix_Solve.topRows(NBins).transpose() *
                          FluxMatrix_Solve.topRows(NBins)) +
                         FluxMatrix_Solve.bottomRows(NFluxes).transpose() *
                             FluxMatrix_Solve.bottomRows(NFluxes))
                            .inverse() *
                        FluxMatrix_Solve.topRows(NBins).transpose() *
                        Target.topRows(NBins);
      } else {
        last_solution = (FluxMatrix_Solve.topRows(NBins).transpose() *
                         FluxMatrix_Solve.topRows(NBins))
                            .inverse() *
                        FluxMatrix_Solve.topRows(NBins).transpose() *
                        Target.topRows(NBins);
      }
      break;
    }
    }

    if (!last_solution.rows()) {
      res_norm = 0;
      soln_norm = 0;
      return last_solution;
    }

    res_norm = ((FluxMatrix_Solve.topRows(NBins) * last_solution) -
                Target.topRows(NBins))
                   .squaredNorm();
    soln_norm = 0;
    if (reg_param > 0) {
      soln_norm =
          (FluxMatrix_Solve.bottomRows(NFluxes) * last_solution / reg_param)
              .squaredNorm();
    }

    return last_solution;
  }
  Eigen::VectorXd Solve(double reg_param = 0, double BC_param = 1) {
    double dum1, dum2;
    std::cout << "reg_param = " << reg_param << std::endl; 
    std::cout << "BC_param = " << BC_param << std::endl; 
    return Solve(reg_param, BC_param, dum1, dum2);
  }

  void Write(TDirectory *td) {
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
    Target_rescale.bottomRows(NBins - FitIdxHigh).array() /= fParams.OORFactor;
    FillHistFromEigenVector(FDFlux_targ, Target_rescale, low_offset);

    TH1 *FDFlux_bf = static_cast<TH1 *>(FDFlux_osc->Clone("FDFlux_bf"));
    FDFlux_bf->Reset();

    Eigen::VectorXd bf = (FluxMatrix_Full * last_solution);
    FillHistFromEigenVector(FDFlux_bf, bf, low_offset);

    TH1D *Coeffs =
        new TH1D("Coeffs", "", last_solution.rows(), 0, last_solution.rows());
    FillHistFromEigenVector(Coeffs, last_solution);

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
//    BCTree->SetDirectory(td);
  }
};

#ifdef FLS_WRAP_IN_NAMESPACE
}
#endif

#endif
