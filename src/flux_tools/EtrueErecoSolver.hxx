#ifndef ETRUEERECO_HXX_SEEN
#define ETRUEERECO_HXX_SEEN

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
#include "TF1.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TLine.h"
#include "TMath.h"
#include "TRandom.h"

#include "Math/Minimizer.h"
//#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/GSLMinimizer.h"
#include "Minuit2/Minuit2Minimizer.h"

#include <time.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#ifdef ERS_WRAP_IN_NAMESPACE
namespace ers {
#endif

double FrobeniusNorm( const Eigen::MatrixXd & theMatrix ) {
  double norm = 0;
  for (int row_it = 0; row_it < theMatrix.rows(); row_it++) {
    for (int col_it = 0; col_it < theMatrix.cols(); col_it++) {
      norm += std::abs( pow( theMatrix(row_it,col_it), 2) );
    }
  }
  norm = std::sqrt(norm);
  return norm;
}

class TheFitFunction { 

public:

  //Constructor 
  TheFitFunction( const Eigen::MatrixXd &atruefluxmat, const Eigen::MatrixXd &arecofluxmat) { 
    // any extra variable stuff here
    aTrueFluxMat = atruefluxmat;
    aRecoFluxMat = arecofluxmat;
    Ebins = aTrueFluxMat.rows();
  }

  // Tracking Ebins..
  Eigen::MatrixXd aTrueFluxMat;
  Eigen::MatrixXd aRecoFluxMat;
  int Ebins;
  

  Double_t operator()(const double* par ) {

    Eigen::MatrixXd tmpMat = Eigen::MatrixXd::Zero(Ebins,Ebins); 
    for (int i = 0; i < pow(Ebins,2); i++) { 
      // std::cout << "Ebins " << Ebins << ", i " << i << std::endl;
      // std::cout << std::remainder(i, Ebins) << "," << (i/Ebins) << std::endl;
      // std::cout << i%Ebins << "," << (i/Ebins) << std::endl;
      tmpMat(i%Ebins, (i/Ebins)) = par[i];
    }

    Eigen::MatrixXd resultMat = aRecoFluxMat - (tmpMat * aTrueFluxMat); 
    double norm = FrobeniusNorm( resultMat ) ;

    return norm;

  }

};


class ERecoSolver {

public:
  struct Params {
    enum Solver { kSVD = 1, kQR, kNormal, kInverse, kCOD, kConjugateGradient, kLeastSquaresConjugateGradient, kBiCGSTAB, kBiCGSTABguess };
    enum ToyMatrixType { mRandom, mRandomLimited, mGauss }; 
    enum ToyFluxType { fRandom, fRandomLimited, fGauss }; 
    enum SmearType { sRandom, sRandomLimited, sGauss }; 
    enum ComputeType { cMatrixMap }; 

    Solver algo_id;
    ToyMatrixType toyM_id;
    ToyFluxType toyF_id;
    SmearType smear_id;
    ComputeType comp_id;

    int nEbins;

    bool scalebyE;
    bool scaleOrderOne;

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
    size_t LeastNCoeffs;

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

    std::pair<size_t, double> NominalFlux;

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
    p.toyM_id = Params::mRandom;
    p.toyF_id = Params::fRandom;
    p.smear_id = Params::sRandom;
    p.comp_id = Params::cMatrixMap;
    p.nEbins = 10;
    p.scalebyE = false;
    p.scaleOrderOne = false;
    p.OORMode = Params::kGaussianDecay;
    p.OORSide = Params::kLeft;
    p.OORFactor = 0.1;
    p.FitBetweenFoundPeaks = true;
    p.MergeENuBins = 0;
    p.MergeOAPBins = 0;
    p.OffAxisRangesDescriptor = "-1.45_37.45:0.1";
    p.NominalOffAxisRangesDescriptor = "-1.45_37.45:0.1";
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

  // Internal Global Variables to be passed around
  Eigen::MatrixXd FullBinTrueSensingMatrix;
  
  Eigen::MatrixXd TrueSensingMatrix;
  Eigen::MatrixXd SmearedSensingMatrix;
  Eigen::MatrixXd RecoSensingMatrix;

  Eigen::MatrixXd SquareTrueFluxMatrix;
  Eigen::MatrixXd TrueFluxMatrix;
  Eigen::MatrixXd RecoFluxMatrix;
  Eigen::MatrixXd SmearedRecoFluxMatrix;
  Eigen::MatrixXd RestoredTrueFluxMatrix;

  // TrueFluxMatrix is all OA fluxes but rebinned in E, FluxMatrix_Full is all unrebinned fluxes
  
  Eigen::MatrixXd FluxMatrix_Full;
  Eigen::MatrixXd FluxMatrix_zSlice;

  Eigen::VectorXd FDFluxVector;
  Eigen::VectorXd FDFluxVectorRebinned;

  std::unique_ptr<TH1> FDFluxHist; 
  std::unique_ptr<TH1> FDFluxOriginal; 

  size_t NCoefficients;
  std::vector<size_t> nZbins;
  std::vector<std::vector<size_t>> AllZbins;
  std::vector<std::vector<size_t>> AllOAbins;
  std::vector<int> OAbinsperhist;
  std::vector<std::vector<double>> zCenter;
  size_t NomRegFluxFirst = 0, NomRegFluxLast = 0;



  void Initialize(Params const &p,
                  std::pair<std::string, std::vector<std::string>> NDFluxDescriptor = {"", {""} },
                  std::pair<std::string, std::string> FDFluxDescriptor = {"", {""} },
		  int Ebins = 10) {
    fParams = p;

    // probably do some loading of actual fluxes here
    if ( NDFluxDescriptor.first.size() && NDFluxDescriptor.second.size() ) {
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
      SetNDFluxes(std::move(Flux3DList), Ebins);
    }

    if (FDFluxDescriptor.first.size() && FDFluxDescriptor.second.size()) {
      std::unique_ptr<TH1> fdflux =
          GetHistogram<TH1>(FDFluxDescriptor.first, FDFluxDescriptor.second);
      SetFDFlux(fdflux.get(), Ebins);
    }
    std::cout << "Initialized" << std::endl; 
  }

  void SetNDFluxes(std::vector<std::unique_ptr<TH3>> const &NDFluxes, int MergedEbins, bool ApplyXRanges = true) {

    std::vector<std::pair<double, double>> XRanges;
    if (ApplyXRanges && fParams.OffAxisRangesDescriptor.size()) {
      XRanges = BuildRangesList(fParams.OffAxisRangesDescriptor);
    }

    std::vector<std::pair<double, double>> NomXRanges;
    if (ApplyXRanges && fParams.NominalOffAxisRangesDescriptor.size()) {
      NomXRanges = BuildRangesList(fParams.NominalOffAxisRangesDescriptor);
    }

    if ( ! fParams.CurrentRangeSplit.size() ) {
      std::vector<std::pair<double, double>> tmpvec{{fParams.CurrentRange.first, fParams.CurrentRange.second}};
      fParams.CurrentRangeSplit.assign(NDFluxes.size(), tmpvec);
    }

    std::vector<int> EbinsVec;
    EbinsVec.emplace_back(MergedEbins);
    EbinsVec.emplace_back(0);
    for (int Ebins : EbinsVec) {
      std::vector<Eigen::MatrixXd> NDMatrices;
      zCenter.clear();
      AllZbins.clear();
      AllOAbins.clear();
      nZbins.clear();
      NomRegFluxFirst = 0;
      NomRegFluxLast = 0;
      for (size_t Hist3D_it = 0; Hist3D_it < NDFluxes.size(); Hist3D_it++ ) {
        std::unique_ptr<TH3> Flux3D(static_cast<TH3 *>(NDFluxes[Hist3D_it]->Clone()));
        Flux3D->SetDirectory(nullptr);
  
        std::vector<std::pair<double, double>> HistCurrents = fParams.CurrentRangeSplit[Hist3D_it];
        std::vector<size_t> HistZbins;
        for (int zbi_it = 1; zbi_it <= Flux3D->GetZaxis()->GetNbins(); zbi_it++) {
          // if high edge is higher than first val and low edge is lower than second val
          for (size_t ranges_it = 0; ranges_it < HistCurrents.size(); ranges_it++) {
            if ( Flux3D->GetZaxis()->GetBinUpEdge(zbi_it) > HistCurrents[ranges_it].first &&
  		 Flux3D->GetZaxis()->GetBinLowEdge(zbi_it) < HistCurrents[ranges_it].second ) {
  	      HistZbins.emplace_back(zbi_it);
  	    }
  	  }
        }
        AllZbins.emplace_back(HistZbins);
  
        size_t NominalZbin = Flux3D->GetZaxis()->FindFixBin( fParams.NominalFlux.second );
  
        int zBins = HistZbins.size(); 
        nZbins.emplace_back(zBins);
        std::vector<size_t> nOAbins;
  
        std::vector<double> tmpv;
        for (size_t zbi_it : HistZbins) {
  	  tmpv.emplace_back(Flux3D->GetZaxis()->GetBinCenter(zbi_it)); 
        }
        zCenter.emplace_back(tmpv);
        
        for ( size_t z : HistZbins ) {
            Flux3D->GetZaxis()->SetRange(z,z);
            TH2 *projectedFlux = (TH2*)Flux3D->Project3D("yx");
        	  std::unique_ptr<TH2> Flux2D(static_cast<TH2 *>(projectedFlux->Clone()));
        	  Flux2D->SetDirectory(nullptr);
  	  if (fParams.scaleOrderOne) {
  	    Flux2D->Scale(1E8);
  	  }
  
  	  if (fParams.scalebyE) {
  	    for (int y_it = 0; y_it < Flux2D->GetYaxis()->GetNbins(); y_it++) {
  	      for (int x_it = 0; x_it < Flux2D->GetXaxis()->GetNbins(); x_it++) {
  	        Flux2D->SetBinContent(x_it, y_it, (Flux2D->GetBinContent(x_it,y_it))*(Flux2D->GetXaxis()->GetBinCenter(x_it)) );
  	      }
  	    }
  	  }
  
  	  // Rebin to get Ebins in X and the requested merged bins in Y 
          if (Ebins && fParams.MergeOAPBins) {
              int MergeX = Flux2D->GetXaxis()->GetNbins()/Ebins;
              int MergeY = fParams.MergeOAPBins; 
              Flux2D->Rebin2D(MergeX, MergeY);
              Flux2D->Scale(1.0 / double(MergeX * MergeY));
  	  } else if (Ebins) {
              int MergeX = Flux2D->GetXaxis()->GetNbins()/Ebins;
              Flux2D->RebinX(MergeX);
              Flux2D->Scale(1.0 / double(MergeX));
          } else if (fParams.MergeOAPBins) {
              Flux2D->RebinY(fParams.MergeOAPBins);
              Flux2D->Scale(1.0 / double(fParams.MergeOAPBins));
  	  }
  
            /*if (Ebins && fParams.MergeOAPBins) {
              Flux2D->Rebin2D(Ebins, fParams.MergeOAPBins);
              Flux2D->Scale(1.0 / double(Ebins * fParams.MergeOAPBins));
            } else if (Ebins) {
              Flux2D->RebinX(Ebins);
              Flux2D->Scale(1.0 / double(Ebins));
            } else if (fParams.MergeOAPBins) {
              Flux2D->RebinY(fParams.MergeOAPBins);
              Flux2D->Scale(1.0 / double(fParams.MergeOAPBins));
            }*/
  
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
  	  // std::cout << "nOAbins[z] : " << FluxMatrix_zSlice.cols() << std::endl;
  	  // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  	  // std::cout << FluxMatrix_zSlice.format(CleanFmt) << std::endl;
  
  	
  	  // Setting TrueFluxMatrix to square matrix form
  	  // std::cout << "z : " << z << std::endl; 
  	  if (Ebins) {
  	    std::cout << "Getting SquareTrueFluxMatrix as integrated square matrix form" << std::endl;
              /*int MergeX = Flux2D->GetXaxis()->GetNbins()/Ebins;
              int MergeY = Flux2D->GetYaxis()->GetNbins()/Ebins;
  	    std::pair<int, int> xLimits;
  	    std::pair<int, int> yLimits;
  
              Flux2D->Rebin2D(MergeX, MergeY);
              Flux2D->Scale(1.0 / double(MergeX * MergeY));
  	    */
  
              int MergeY = Flux2D->GetYaxis()->GetNbins()/Ebins;
  	    std::cout << "Matrix contains.. " << std::endl;
  	    std::cout << "Xbins : " << Flux2D->GetXaxis()->GetNbins() << std::endl;
  	    std::cout << "Ybins : " << Flux2D->GetYaxis()->GetNbins() << std::endl;
              Flux2D->RebinY(MergeY);
              Flux2D->Scale(1.0 / double(MergeY));
  
  	    std::pair<int, int> xLimits;
  	    std::pair<int, int> yLimits;
  	    int xb = Flux2D->GetXaxis()->GetNbins();
  	    int yb = Flux2D->GetYaxis()->GetNbins();
  	    // Choose the smaller of two axes
  	    /*xb = std::min(xb, yb);
  	    yb = std::min(xb, yb);*/ // removed because of errors for large Ebins further on..
    	    xLimits = { Flux2D->GetXaxis()->GetBinCenter(1), Flux2D->GetXaxis()->GetBinCenter(xb) };
  	    // yLimits = { Flux2D->GetYaxis()->GetBinCenter(1), Flux2D->GetYaxis()->GetBinCenter(yb) };
  	    yLimits = { Flux2D->GetYaxis()->GetBinCenter(1), Flux2D->GetYaxis()->GetBinCenter(xb) };
  	    std::cout << "Constructing square matrix between.. " << std::endl; 
  	    std::cout << " " << xLimits.first << "," << xLimits.second << " GeV" << std::endl;
  	    std::cout << " " << yLimits.first << "," << yLimits.second << " m OA" << std::endl;
  
  	    // SquareTrueFluxMatrix = Eigen::MatrixXd::Zero( xb, yb);
  	    SquareTrueFluxMatrix = Eigen::MatrixXd::Zero( xb, xb);
  
  	    // for every y bin from lim0 to lim1
  	    // for every x bin from lim0 to lim1
  	    if (xb < yb) {
	      yb = xb;
	    }
  	    for (int ybin = 0; ybin < yb; ybin++) {
  	      // std::unique_ptr<TH1> split; 
  	      TH1* split; 
  	      split = Flux2D->ProjectionX( (to_str(ybin+1) + "_to_" + to_str(ybin+1)).c_str(), ybin + 1, ybin + 1);
                for (int xbin = 0; xbin < xb; xbin++) {
                  SquareTrueFluxMatrix(xbin, ybin) = split->GetBinContent(xbin  + 1);
                }
  	    }
  	    if (xb > yb) {
  	      int extrabins = xb - yb;
        	      std::unique_ptr<TH2> tmpFlux2D(static_cast<TH2 *>(projectedFlux->Clone()));
  	      int nXbins = tmpFlux2D->GetXaxis()->GetNbins();
                int MergeX = tmpFlux2D->GetXaxis()->GetNbins()/Ebins;
                tmpFlux2D->RebinX(MergeX);
                tmpFlux2D->Scale(1.0 / double(MergeX));
  	      for (int ex_it = 0; ex_it < extrabins; ex_it++) {
  	        int ybin = yb + ex_it;
  	      TH1  * split; 
  	        split = tmpFlux2D->ProjectionX( (to_str(ybin+1) + "_to_" + to_str(ybin+1)).c_str(), ybin + 1, ybin + 1);
                  for (int xbin = 0; xbin < xb; xbin++) {
                    SquareTrueFluxMatrix(xbin, ybin) = split->GetBinContent(xbin  + 1);
  	        }
  	      }
  	    }
  
  	    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  	    std::cout << " --- Square True Flux Matrix --- " << std::endl;
  	    std::cout << SquareTrueFluxMatrix.format(CleanFmt) << std::endl;
  
  	  }
        }
  
        AllOAbins.emplace_back(nOAbins);
  
        if ( Hist3D_it == (fParams.NominalFlux.first - 1) ) {
  	// runs through all previous 3D histograms
          for (size_t prevhist_it = 0; prevhist_it < Hist3D_it; prevhist_it++) {
  	  std::cout << " nZbins["<<prevhist_it<<"] : " <<  nZbins[prevhist_it] << std::endl; 
      	  for (size_t OAbins_it = 0; OAbins_it < nZbins[prevhist_it]; OAbins_it++) {
    	    NomRegFluxFirst += AllOAbins[prevhist_it][OAbins_it];
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
  	size_t index = std::distance(AllZbins[Hist3D_it].begin(), it);
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
      Int_t FullMcols = 0;
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
  
      Eigen::MatrixXd tmpMat = Eigen::MatrixXd(NDrows, FullMcols);
      /// TrueFluxMatrix = Eigen::MatrixXd(NDrows, FullMcols);
      for (size_t n = 0; n < NDMatrices.size(); n++) {
        // Int_t prevcols = all NDcols for m < n
        Int_t prevcols = 0;
        for (size_t prev_it = 0; prev_it < n; prev_it++) {
          prevcols += NDMatrices[prev_it].cols();
        }
        tmpMat.block(0, 0 + prevcols, NDrows, NDMatrices[n].cols()) = NDMatrices[n];
        /// TrueFluxMatrix.block(0, 0 + prevcols, NDrows, NDMatrices[n].cols()) = NDMatrices[n];
      }
      // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
      // std::cout << FluxMatrix_Full.format(CleanFmt) << std::endl;
      //
      if (Ebins) {
        TrueFluxMatrix = tmpMat;
      /// NCoefficients = TrueFluxMatrix.cols();
      } else {
        FluxMatrix_Full = tmpMat;
        NCoefficients = FluxMatrix_Full.cols();
      }
    }
  }

  void SetFDFlux(TH1 *const fdflux, int Ebins = 0) {
    FDFluxHist = std::unique_ptr<TH1>(static_cast<TH1 *>(fdflux->Clone()));
    FDFluxHist->SetDirectory(nullptr);

    int originalBins = FDFluxHist->GetXaxis()->GetNbins();
    FDFluxVector = Eigen::VectorXd::Zero(originalBins);
    for (int bi_it = 0; bi_it < originalBins; bi_it++) {
      FDFluxVector(bi_it) = FDFluxHist->GetBinContent(bi_it+1);
    }

    if (Ebins) {
      int MergeX = FDFluxHist->GetXaxis()->GetNbins()/Ebins;
      FDFluxHist->Rebin(MergeX);
      FDFluxHist->Scale(1.0 / double(MergeX));
    }

    FDFluxVectorRebinned = Eigen::VectorXd::Zero(Ebins);
    for (int bi_it = 0; bi_it < Ebins; bi_it++) {
      FDFluxVectorRebinned(bi_it) = FDFluxHist->GetBinContent(bi_it+1);
    }
    /*if (fParams.scaleOrderOne) {
      FDFluxVector *= 1E8;
    }*/
    FDFluxOriginal = std::unique_ptr<TH1>(static_cast<TH1 *>(fdflux->Clone()));
    FDFluxOriginal->SetDirectory(nullptr);

  }

  Eigen::VectorXd VectorRebin(Eigen::VectorXd aVec, int newbins, bool rescale = true ) {
    int oldbins = aVec.size();
    int mergedbins = oldbins/newbins; 

    Eigen::VectorXd newVec = Eigen::VectorXd::Zero(newbins);

    if (!mergedbins) {
	std::cout << " ---------- [ERROR] ---------- " << std::endl;
	std::cout << "Rebinned vector only has " << oldbins << " bins, asked to rebin for " << newbins << " new bins " << std::endl;
	return newVec;
    }

    for (int bin_it = 0; bin_it < newbins; bin_it++) {
      double mergedbin = 0;
      for (int merge_it = 0; merge_it < mergedbins; merge_it++) {
        mergedbin += aVec( (bin_it*mergedbins) + merge_it);
      }
      if (rescale) {
        newVec(bin_it) = mergedbin/mergedbins;
      } else {
        newVec(bin_it) = mergedbin;
      }
    }

    return newVec;
  }

  Eigen::MatrixXd MatrixRebinRows(Eigen::MatrixXd aMat, int newrows, bool rescale = true ) {
    int oldrows = aMat.rows();
    int mergedrows = oldrows/newrows; 

    Eigen::MatrixXd newMat = Eigen::MatrixXd::Zero(newrows, aMat.cols());

    if (!mergedrows) {
	std::cout << " ---------- [ERROR] ---------- " << std::endl;
	std::cout << "Rebinned matrix only has " << oldrows << " rows, asked to rebin for " << newrows << " new rows " << std::endl;
	return newMat;
    }

    for (int col_it = 0; col_it < aMat.cols(); col_it++) {
      for (int row_it = 0; row_it < newrows; row_it++) {
        double mergedbin = 0;
        for (int merge_it = 0; merge_it < mergedrows; merge_it++) {
	  mergedbin += aMat( (row_it*mergedrows) + merge_it, col_it );
	}
	if (rescale) {
	  newMat(row_it, col_it) = mergedbin/mergedrows;
	} else {
	  newMat(row_it, col_it) = mergedbin;
	}
      }
    }

    return newMat;
  }

  Eigen::MatrixXd MatrixRebinCols(Eigen::MatrixXd aMat, int newcols, bool rescale = true ) {
    int oldcols = aMat.cols();
    int mergedcols = oldcols/newcols; 

    Eigen::MatrixXd newMat = Eigen::MatrixXd::Zero(aMat.rows(), newcols );

    if (!mergedcols) {
	std::cout << "[ERROR]" << std::endl;
	std::cout << "Rebinned matrix only has " << oldcols << " columns, asked to rebin for " << newcols << " new cols " << std::endl;
	return newMat;
    }

    for (int row_it = 0; row_it < aMat.rows(); row_it++) {
      for (int col_it = 0; col_it < newcols; col_it++) {
        double mergedbin = 0;
        for (int merge_it = 0; merge_it < mergedcols; merge_it++) {
	  mergedbin += aMat( row_it, (col_it*mergedcols) + merge_it );
	}
	if (rescale) {
	  newMat(row_it, col_it) = mergedbin/mergedcols;
	} else {
	  newMat(row_it, col_it) = mergedbin;
	}
      }
    }

    return newMat;
  }

  void doMatrixMapAnalysis(int Ebins, int NFluxes, double SenseSmearingLimit, double SSLpb, double NoiseSmearingLimit, bool SmearSensingMatrix, bool SmearRecoFlux, std::string OutputFile) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    // TrueFluxMatrix = SquareTrueFluxMatrix;
    Ebins = TrueFluxMatrix.rows();
    std::cout << "TrueFluxMatrix.rows() = " << Ebins << std::endl;
    std::cout << "FluxMatrix_Full.rows() = " << FluxMatrix_Full.rows() << std::endl;
    if ( true == false ) {
      ///LoadToySensingMatrix(Ebins);
      ///RecoFluxMatrix = TrueSensingMatrix * SquareTrueFluxMatrix;
    }

    LoadToySensingMatrix(FluxMatrix_Full.rows(), SenseSmearingLimit, SSLpb);
    RecoFluxMatrix = TrueSensingMatrix * FluxMatrix_Full; 
    // std::cout << RecoFluxMatrix.format(CleanFmt) << std::endl; 

    /////RecoFluxMatrix = TrueSensingMatrix * SquareTrueFluxMatrix;
    
    // std::cout << " --- Reco Fluxes --- " << std::endl;
    // std::cout << RecoFluxMatrix.format(CleanFmt) << std::endl; 

    if (SmearRecoFlux) {
      std::cout << "Smear Reco Fluxes" << std::endl;
      SmearedRecoFluxMatrix = SmearMatrix(RecoFluxMatrix, NoiseSmearingLimit);
      std::cout << " --- Smeared Reco Fluxes --- " << std::endl;
      std::cout << SmearedRecoFluxMatrix.format(CleanFmt) << std::endl;
    } else {
      SmearedRecoFluxMatrix = RecoFluxMatrix;
    }

    Eigen::MatrixXd RebinnedSmearedRecoFluxMatrix = MatrixRebinRows(SmearedRecoFluxMatrix, Ebins);
    std::cout << " --- Rebin 1 : Reco Fluxes --- " << std::endl;
    std::cout << RebinnedSmearedRecoFluxMatrix.format(CleanFmt) << std::endl; 
    Eigen::MatrixXd SquareSmearedRecoFluxMatrix = MatrixRebinCols(RebinnedSmearedRecoFluxMatrix, Ebins);
    std::cout << " --- Rebin 2 : Reco Fluxes --- " << std::endl;
    std::cout << SquareSmearedRecoFluxMatrix.format(CleanFmt) << std::endl; 


    ComputeMatrix(SquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix);
    // RestoredTrueFluxMatrix = RecoSensingMatrix.inverse() * SmearedRecoFluxMatrix;// Only works if you change them to square fluxes

    // Fit sensing matrix with TrueFluxMatrix - set to square starting mats
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(SquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix);

    // Fit sensing matrix with TrueFluxMatrix : rebin rows to get Ebins as required 
    Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix( MatrixRebinRows(TrueFluxMatrix, Ebins), RebinnedSmearedRecoFluxMatrix, true);

    // Fit sensing matrix with FluxMatrix_Full - all fluxes
    std::cout << TrueSensingMatrix.cols() << "," << TrueSensingMatrix.rows() << std::endl;
    std::cout << FluxMatrix_Full.cols() << "," << FluxMatrix_Full.rows() << std::endl;
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueFluxMatrix, (TrueSensingMatrix*TrueFluxMatrix) );

    /*
    // Set negative values to positive small values & fit with remaining dataset 
    Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(Ebins, NFluxes);
    Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix;

    Eigen::MatrixXd SmearedRecoToyFluxMatrix = SmearMatrix(RecoToyFluxMatrix, SmearingLimit);
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueToyFluxMatrix, RecoToyFluxMatrix);
    Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueToyFluxMatrix, SmearedRecoToyFluxMatrix);
    // Eigen::MatrixXd fittedSensingMatrix = RecoSensingMatrix;
    */
    std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
    std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
    std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
    std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
    //////////////// Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), FluxMatrix_Full.rows());
    /*Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, FluxMatrix_Full.rows());
    std::cout << " --- Scaled Up Fitted Sensing Matrix --- " << std::endl;
    std::cout << "ScaledUpFittedSensingMatrix.rows() :" << ScaledUpFittedSensingMatrix.rows() << std::endl;
    std::cout << "ScaledUpFittedSensingMatrix.cols() :" << ScaledUpFittedSensingMatrix.cols() << std::endl;
    std::cout << " --- True Sensing Matrix --- " << std::endl;
    std::cout << "TrueSensingMatrix.rows() :" << TrueSensingMatrix.rows() << std::endl;
    std::cout << "TrueSensingMatrix.cols() :" << TrueSensingMatrix.cols() << std::endl;*/
    // std::cout << ScaledUpFittedSensingMatrix.format(CleanFmt) << std::endl;

    // std::cout << " --- True Sensing Matrix --- " << std::endl;
    // std::cout << TrueSensingMatrix.format(CleanFmt) << std::endl;

    
    
    // Compare With Toys
    /*
    Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(Ebins, NFluxes);
    Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix;
    // ^ moved up
    Eigen::MatrixXd SmearedRecoToyFluxMatrix = SmearMatrix(RecoToyFluxMatrix, SmearingLimit);
    Eigen::MatrixXd RestoredTrueToyFluxMatrix = fittedSensingMatrix.inverse() * SmearedRecoToyFluxMatrix ;
    // Eigen::MatrixXd RestoredTrueToyFluxMatrix = RecoSensingMatrix.inverse() * SmearedRecoToyFluxMatrix ;
    */
    TFile *f = CheckOpenFile(OutputFile, "RECREATE");/*
    WriteMatrix(f, Ebins, TrueToyFluxMatrix, "TrueToyFlux");
    WriteMatrix(f, Ebins, RecoToyFluxMatrix, "RecoToyFlux");
    WriteMatrix(f, Ebins, SmearedRecoToyFluxMatrix, "SmearedRecoToyFlux");
    WriteMatrix(f, Ebins, RestoredTrueToyFluxMatrix, "RestoredTrueToyFlux");*/

    if ( FDFluxVector.size() ) {
      // std::cout << FDFluxVector << std::endl;
      Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVector;
	std::cout << "DEBUG" << std::endl;
      Eigen::VectorXd FDRestoredVector = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);
	std::cout << "DEBUG" << std::endl;
      // std::cout << FDRecoVector << std::endl;
      // Eigen::VectorXd FDRestoredVector = (fittedSensingMatrix.inverse())*VectorRebin(FDRecoVector, Ebins, false); //old - doesnt account for scaledu p size
      // Eigen::VectorXd FDRestoredVector = (ScaledUpFittedSensingMatrix.inverse())*VectorRebin(FDRecoVector, Ebins, false);
      // Eigen::MatrixXd SUFSM = (ScaledUpFittedSensingMatrix.inverse());
      // std::cout << SUFSM.format(CleanFmt) << std::endl;
      // Eigen::VectorXd FDRestoredVector = (ScaledUpFittedSensingMatrix.inverse())*FDRecoVector;
      ///////////////////////////////Eigen::VectorXd FDRestoredVector = ScaledUpFittedSensingMatrixInv*FDRecoVector;
      // std::cout << FDRestoredVector << std::endl;

      /*
      WriteVector(f, Ebins, FDFluxVector, "FDTrueVector", FDFluxHist.get());
      WriteVector(f, Ebins, FDRecoVector, "FDRecoVector", FDFluxHist.get());
      WriteVector(f, Ebins, FDRestoredVector, "FDRestoredVector", FDFluxHist.get());
      */

      WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
      WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVector.size(), FDRestoredVector, "FDRestoredVector", FDFluxOriginal.get());

      // FDFluxHist->SetDirectory(f);
      // FDFluxHist->Write();
      FDFluxOriginal->SetName("FDFluxOriginal");
      FDFluxOriginal->Write();
    }

    // Write1D(f, Ebins);
    f->Close();
  
  }

  void doToyMatrixMapAnalysis(int Ebins, int NFluxes, double SenseSmearingLimit, double NoiseSmearingLimit, bool SmearSensingMatrix, bool SmearRecoFlux) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    LoadToySensingMatrix(Ebins);

    if (SmearSensingMatrix) {
      std::cout << "Smear Sensing Matrix" << std::endl;
      SmearedSensingMatrix = SmearMatrix(TrueSensingMatrix, NoiseSmearingLimit);
      std::cout << " --- Smeared Sensing Matrix --- " << std::endl;
      std::cout << SmearedSensingMatrix.format(CleanFmt) << std::endl;
    } else {
      SmearedSensingMatrix = TrueSensingMatrix;
    }

    TrueFluxMatrix = LoadToyFluxes(Ebins, NFluxes);

    RecoFluxMatrix = TrueSensingMatrix * TrueFluxMatrix;
    std::cout << " --- Reco Fluxes --- " << std::endl;
    std::cout << RecoFluxMatrix.format(CleanFmt) << std::endl;

    if (SmearRecoFlux) {
      std::cout << "Smear Reco Fluxes" << std::endl;
      SmearedRecoFluxMatrix = SmearMatrix(RecoFluxMatrix, NoiseSmearingLimit);
      std::cout << " --- Smeared Reco Fluxes --- " << std::endl;
      std::cout << SmearedRecoFluxMatrix.format(CleanFmt) << std::endl;
    } else {
      SmearedRecoFluxMatrix = RecoFluxMatrix;
    }

    ComputeMatrix(TrueFluxMatrix, SmearedRecoFluxMatrix);
    RestoredTrueFluxMatrix = RecoSensingMatrix.inverse() * SmearedRecoFluxMatrix ;
  }

  void LoadToySensingMatrix(int nEbins, double lim = 0.2, double binlim = 0.01) {
    std::cout << "Load Toy Matrix" << std::endl;
    TrueSensingMatrix = Eigen::MatrixXd::Zero(nEbins, nEbins);

    if (fParams.toyM_id == Params::mRandom) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int row_it = 0; row_it < nEbins; row_it++) {
	for (int col_it = 0; col_it < nEbins; col_it++) {
	  float randomf = (float) rand()/RAND_MAX; 
          TrueSensingMatrix(row_it, col_it) = randomf;
	}
      }
    }

    if (fParams.toyM_id == Params::mRandomLimited) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int col_it = 0; col_it < nEbins; col_it++) {
        // float limit = 0.2;
        float limit = lim;
        TrueSensingMatrix(col_it, col_it) = 1;
	int aGevBins = nEbins/10 + (nEbins%10 != 0) ;
        for (int sub_it = col_it+nEbins/10; sub_it >= 0; sub_it--) {
          if ( sub_it == col_it || sub_it >= nEbins) {
	    continue;
	  }
	  float randomf = (float) rand()/RAND_MAX; 
          // float limrandom = limit*randomf*(0.5*10/nEbins);// up to 50% loss per Gev if limit = 1 
          // float limrandom = randomf * (0.5*10/nEbins);
          float limrandom = randomf * binlim * (limit/lim);
	  // std::cout << "limrandom " << limrandom << std::endl;
	  // std::cout << "limit left " << limit << std::endl;
	  limit -= limrandom;
	  if ( limit <=0 ) {
	    break;
          }
	  TrueSensingMatrix(sub_it, col_it) = limrandom; 
          TrueSensingMatrix(col_it, col_it) -= limrandom;
        }
      }
    }/* else if ( fParams.toyM_id == Params::mGauss ) {
      gRandom->SetSeed(0);
      for (int col_it = 0; col_it < nEbins; col_it++) {
        // float limit = 0.2;
        float limit = lim;
        TrueSensingMatrix(col_it, col_it) = 1;
	int aGevBins = nEbins/10 + (nEbins%10 != 0) ;
        for (int sub_it = col_it+nEbins/10; sub_it >= 0; sub_it--) {
          if ( sub_it == col_it || sub_it >= nEbins) {
	    continue;
	  }
          TString gstr = TString::Format("TMath::Gaus(x,0.5,%f)", lim);
          TF1* fgaus = new TF1("fgaus", gstr.Data(), 0.0, 1.0);
	}
      }
      for (int row_it = 0; row_it < SmearedMatrix.rows(); row_it++) {
        for (int col_it = 0; col_it < SmearedMatrix.cols(); col_it++) {
	  float gausrandom = fgaus->GetRandom();
	  SmearedMatrix(row_it, col_it) *= gausrandom; 
	}
      }
    }*/

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << " --- True Sensing Matrix --- " << std::endl; 
    std::cout << TrueSensingMatrix.format(CleanFmt) << std::endl; 
  }

  Eigen::MatrixXd LoadToyFluxes(int nEbins, int NFluxes) {
    std::cout << "Load Toy Fluxes" << std::endl;
    Eigen::MatrixXd ToyFluxes = Eigen::MatrixXd::Zero(nEbins, NFluxes);

    if (fParams.toyF_id == Params::fRandom) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int row_it = 0; row_it < nEbins; row_it++) {
        for (int col_it = 0; col_it < nEbins; col_it++) {
	  float randomf = (float) rand()/RAND_MAX; 
          ToyFluxes(row_it, col_it) = randomf; 
	}
      }
    }

    if (fParams.toyF_id == Params::fRandomLimited) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int col_it = 0; col_it < nEbins; col_it++) {
        float limit = 0.2;
        ToyFluxes(col_it, col_it) = 1;
        for (int sub_it = col_it+1; sub_it >= 0 && sub_it < nEbins; sub_it--) {
          if ( sub_it == col_it ) {
	    continue;
	  }
	  float randomf = (float) rand()/RAND_MAX; 
          float limrandom = limit*randomf;
	  limit -= limrandom;
	  if ( limit <=0 ) {
	    break;
          }
	  ToyFluxes(sub_it, col_it) = limrandom; 
          ToyFluxes(col_it, col_it) -= limrandom;
        }
      }
    }

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << " --- True Toy Fluxes --- " << std::endl;
    std::cout << ToyFluxes.format(CleanFmt) << std::endl; 

    // RecoFluxMatrix = TrueSensingMatrix * TrueFluxMatrix;
    // std::cout << " --- Reco Fluxes --- " << std::endl;
    // std::cout << RecoFluxMatrix.format(CleanFmt) << std::endl; 

    return ToyFluxes;
  }

  /*void ComputeMatrix() {
    std::cout << "Compute Matrix" << std::endl;

    if (fParams.comp_id == Params::cMatrixMap) {
    Eigen::MatrixXd tmpMatrix = TrueFluxMatrix.inverse();
    RecoSensingMatrix = RecoFluxMatrix * tmpMatrix; 
    }

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << " --- Recovered Sensing Matrix --- " << std::endl;
    std::cout << RecoSensingMatrix.format(CleanFmt) << std::endl; 

  }*/

  Eigen::VectorXd applyLowRankSensingMatrix(Eigen::MatrixXd aMat, Eigen::VectorXd RecoVec) {

    if ( ! RecoVec.size()%aMat.cols()) {
      int multiplier = RecoVec.size()/aMat.cols(); 

      Eigen::VectorXd RestoredVec = Eigen::VectorXd::Zero(RecoVec.size());

      for (int it = 0; it < RecoVec.size(); it++) {
        double newbin = 0;
        for (int sub_it = 0; sub_it < RecoVec.size(); sub_it++) {
	  if (sub_it/multiplier != it/multiplier) {
            newbin += RecoVec(sub_it)*aMat(it/multiplier, sub_it/multiplier)/multiplier;
	  }
	  if (sub_it/multiplier == it/multiplier) {
            newbin += RecoVec(it)*aMat(it/multiplier, it/multiplier)/multiplier;
	  }
        }
        RestoredVec(it) = newbin; 
      }
    } else { 
      // define some averaging between bins over non-divisible binning schema
      
    }
    return RestoredVec;
  }

  Eigen::MatrixXd scaleUpSensingMatrix(Eigen::MatrixXd aMat, int newbins) {
    // want to scale down splittings across rows, but preserve size across columns
    Eigen::MatrixXd newMat1 = Eigen::MatrixXd::Zero(newbins, aMat.cols());
    Eigen::MatrixXd newMat2 = Eigen::MatrixXd::Zero(newbins, newbins);

    Eigen::MatrixXd newMat3 = Eigen::MatrixXd::Zero(aMat.rows(), newbins);

    int multiplier = newbins/aMat.rows() + (newbins%aMat.rows() != 0);

    /*for (int row_it = 0; row_it < aMat.rows(); row_it++) {
      for (int col_it = 0; col_it < aMat.cols(); col_it++) {
        for (int new_it = 0; new_it < multiplier; new_it++) {
	  int bi_it = (col_it*multiplier) + new_it;
	  if ( bi_it >= newbins ) {
	    continue;
	  }
	  // std::cout << row_it <<","<< col_it <<","<< bi_it <<","<< multiplier << std::endl;
	  newMat3(row_it, bi_it) = aMat(row_it, col_it)/multiplier;
	}
      }
    }*/

    for (int col_it = 0; col_it < aMat.cols(); col_it++) {
      for (int row_it = 0; row_it < aMat.rows(); row_it++) {
        for (int new_it = 0; new_it < multiplier; new_it++) {
	  int bi_it = (row_it*multiplier) + new_it;
	  if ( bi_it >= newbins ) {
	    continue;
	  }
	  // std::cout << row_it <<","<< col_it <<","<< bi_it <<","<< multiplier << std::endl;
	  newMat1(bi_it, col_it) = aMat(row_it, col_it)/multiplier;
	}
      }
    }

    for (int row_it = 0; row_it < newMat1.rows(); row_it++) {
      for (int col_it = 0; col_it < newMat1.cols(); col_it++) {
        for (int new_it = 0; new_it < multiplier; new_it++) {
	  int bi_it = (col_it*multiplier) + new_it;
	  if ( bi_it >= newbins ) {
	    continue;
	  }
	  // std::cout << row_it <<","<< col_it <<","<< bi_it <<","<< multiplier << std::endl;
	  newMat2(row_it, bi_it) = newMat1(row_it, col_it);
	}
      }
    }

    return newMat2;
    // return newMat3;
  }

  void ComputeMatrix(const Eigen::MatrixXd &atruefluxMatrix, const Eigen::MatrixXd &arecofluxMatrix ) {

    if (fParams.comp_id == Params::cMatrixMap) {
      Eigen::MatrixXd tmpMatrix = atruefluxMatrix.inverse();
      RecoSensingMatrix = arecofluxMatrix * tmpMatrix;
    }

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << " --- Recovered Sensing Matrix --- " << std::endl;
    std::cout << RecoSensingMatrix.format(CleanFmt) << std::endl;
  }

  void doSmearing( bool SmearSensingMatrix, bool SmearRecoFlux) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    if (SmearSensingMatrix) {
      std::cout << "Smear Sensing Matrix" << std::endl;
      SmearedSensingMatrix = SmearMatrix(TrueSensingMatrix);
      std::cout << " --- Smeared Sensing Matrix --- " << std::endl;
      std::cout << SmearedSensingMatrix.format(CleanFmt) << std::endl;
    }

    if (SmearRecoFlux) {
      std::cout << "Smear Reco Fluxes" << std::endl;
      SmearedRecoFluxMatrix = SmearMatrix(RecoFluxMatrix);
      std::cout << " --- Smear Reco Fluxes --- " << std::endl;
      std::cout << SmearedRecoFluxMatrix.format(CleanFmt) << std::endl;
    }
  }

  Eigen::MatrixXd SmearMatrix( const Eigen::MatrixXd &aMatrix, float SmearingLim = 0.05 ) {

    Eigen::MatrixXd SmearedMatrix = aMatrix;

    srand( time(NULL) ); // Randomize seed initialization 
    // float SmearingLim = 0.05; // Should replace with gaussian..
    if ( fParams.smear_id == Params::sRandom ) {
      for (int row_it = 0; row_it < SmearedMatrix.rows(); row_it++) { 
        for (int col_it = 0; col_it < SmearedMatrix.cols(); col_it++) { 
	  float randomf = (float) rand()/RAND_MAX; 
	  SmearedMatrix(row_it, col_it) *= ((1 - SmearingLim) + (randomf*SmearingLim)); // Should replace with gaussian..
	}
      }
    } else if ( fParams.smear_id == Params::sGauss ) {
      TString gstr = TString::Format("TMath::Gaus(x,1.0,%f)", SmearingLim);
      TF1* fgaus = new TF1("fgaus", gstr.Data(), 0.0, 2.0);
      gRandom->SetSeed(0);
      for (int row_it = 0; row_it < SmearedMatrix.rows(); row_it++) { 
        for (int col_it = 0; col_it < SmearedMatrix.cols(); col_it++) { 
	  float gausrandom = fgaus->GetRandom();
	  SmearedMatrix(row_it, col_it) *= gausrandom; 
	}
      }
    }

    return SmearedMatrix;
  }

  Eigen::MatrixXd fitSensingMatrix( const Eigen::MatrixXd &aTrueFluxMat, const Eigen::MatrixXd &aRecoFluxMat, bool guidematrix = false ) {
    int Ebins = 0;
    if (aTrueFluxMat.cols() != aRecoFluxMat.cols()) {
      std::cout << "[ERROR]:Wrong cols" << std::endl;
      return Eigen::MatrixXd::Zero(1,1); 
    } else {
      Ebins = aTrueFluxMat.rows();
    }

    // set up problem.. root fitter
    // TF1* theFun = new TF1("theFun", myFunction, xMin, xMax, pow(Ebins, 2)); 
    // ROOT::Math::GSLMinimizer min( ROOT::Math::kVectorBFGS );
    // ROOT::Math::GSLMinimizer min(  ROOT::Math::kConjugateFR  );
    ROOT::Minuit2::Minuit2Minimizer min( ROOT::Minuit2::kMigrad  );
    // min.SetPrintLevel(1);

    min.SetMaxFunctionCalls(10000000);
    min.SetMaxIterations(1000000);
    min.SetTolerance(0.000001);

    // ROOT::Math::Functor f(&ERecoSolver::myFunction, Ebins);
    TheFitFunction fObj(aTrueFluxMat, aRecoFluxMat);

    ROOT::Math::Functor f(fObj, pow(Ebins,2) );
    // ROOT::Math::Functor f(&myFunction, pow(Ebins,2) );
    min.SetFunction(f);

    if (guidematrix) {
      for (int diag_it = 0; diag_it < RecoSensingMatrix.cols(); diag_it++) {
        RecoSensingMatrix(diag_it, diag_it) = 0.9;
      }
    }
   
    for (int col_it = 0; col_it < RecoSensingMatrix.cols(); col_it++) {
      for (int row_it = 0; row_it < RecoSensingMatrix.rows(); row_it++) {
	int index = row_it + (col_it*Ebins);
	double val = RecoSensingMatrix(row_it, col_it);
	if ( val < 0 ) {
	  val = 1E-15;
	} else if ( val > 1 ) {
	  val = 0.9;
	}
	// min.SetVariable(index, to_str(index).c_str(), val, 0.000001 );
	min.SetVariable(index, to_str(index).c_str(), val, 0.1 );
	// min.SetVariable(index, to_str(index).c_str(), RecoSensingMatrix(row_it, col_it), 0.01 );
	min.SetVariableLimits(index, 0.0, 1.0);
      }
    }

    std::cout << "The minimizing sensing matrix has size : " << RecoSensingMatrix.rows() << "," << RecoSensingMatrix.cols() << std::endl;
    std::cout << "The minimizing flux matrix has size : " << aTrueFluxMat.rows() << "," << aTrueFluxMat.cols() << std::endl;
    min.Minimize();

    std::cout << " Minimized " << std::endl;
    Eigen::MatrixXd tmpMat = Eigen::MatrixXd::Zero(Ebins, Ebins); 
    const double *xs = min.X();
    for (int i = 0; i < RecoSensingMatrix.rows()*RecoSensingMatrix.cols(); i++) { 
      // tmpMat(std::remainder(i, Ebins), (i/Ebins)) = xs[i];
      tmpMat(i%Ebins, (i/Ebins)) = xs[i];
    }

    double norm = FrobeniusNorm( aRecoFluxMat - (tmpMat * aTrueFluxMat) ) ;
    std::cout << "Frobenius norm of Ereco - (fittedMatrix*Etrue) : " << norm << std::endl; 

    return tmpMat;
  }

  /*Double_t myFunction(Double_t* par) {

    // Get the x value from the first argument array
    // Double_t x = xArr[0];

    // Parameters from the second argument array
    // int Ebins = par.size();
    // int Ebins = RecoSensingMatrix.rows();
    int Ebins = RecoSensingMatrix.cols();
    Eigen::MatrixXd tmpMat = RecoSensingMatrix; 
    for (int i = 0; i < Ebins; i++) { 
      tmpMat(std::remainder(i, Ebins), (i/Ebins)) = par[i];
    }

    // Eigen::MatrixXd resultMat = aRecoFluxMat - (aTrueFluxMat * 

    // Eigen::MatrixXd resultMat = tmpMat.inverse() * aRecoFluxMat;
    // double norm = FrobeniusNorm( (tmpMat.inverse() * aRecoFluxMat) - TrueSensingMatrix );
    double norm = FrobeniusNorm( tmpMat );
    std::cout<<norm<<std::endl;
    return norm;

  }*/

  void WriteVector(TDirectory *td, int nEbins, Eigen::VectorXd aVector, TString prefix, TH1* templateHist = nullptr) {
    TString hname = TString::Format("%s", prefix.Data());
    TH1D *aHist = new TH1D (hname, hname, nEbins, 0.0, 10.0);
    if (templateHist) {
      aHist = static_cast<TH1D*>(templateHist->Clone(hname));
      aHist->Reset();
    }
    FillHistFromEigenVector(aHist, aVector);

    aHist->SetDirectory(td);
    aHist->Write();
  }

  void WriteMatrix(TDirectory *td, int nEbins, Eigen::MatrixXd aMat, TString prefix) {
    for (int flux_it = 0; flux_it < aMat.cols(); flux_it++) {
      TString hname = TString::Format("%s_%i", prefix.Data(), flux_it+1);
      TH1D *aHist = new TH1D (hname, hname, nEbins, 0.0, 10.0);
      Eigen::VectorXd aVector = aMat.col(flux_it);
      FillHistFromEigenVector(aHist, aVector);

      aHist->SetDirectory(td);
      aHist->Write();
    }
  }

  /*void Write1Dtoy(TDirectory *td, int nEbins) {

    for (int flux_it = 0; flux_it < TrueToyFluxMatrix.cols(); flux_it++) {

      TString truefh = TString::Format("TrueToyFlux_%i", flux_it+1);
      TH1D *aHist = new TH1D (truefh, truefh, nEbins, 0.0, 10.0);
      Eigen::VectorXd TrueToyFluxVector = TrueToyFluxMatrix.col(flux_it);
      FillHistFromEigenVector(aHist, TrueToyFluxVector);

      TString recofh = TString::Format("RecoToyFlux_%i", flux_it+1);
      TH1 *bHist = static_cast<TH1 *>(aHist->Clone(recofh));
      bHist->Reset();
      // TH1D *bHist = new TH1D (recofh, recofh, nEbins, 0.0, 10.0);
      Eigen::VectorXd RecoToyFluxVector = RecoToyFluxMatrix.col(flux_it);
      FillHistFromEigenVector(bHist, RecoToyFluxVector);

      TString restfh = TString::Format("RestoredTrueToyFlux_%i", flux_it+1);
      TH1 *cHist = static_cast<TH1 *>(aHist->Clone(restfh));
      cHist->Reset();
      Eigen::VectorXd RestoredTrueToyFluxVector = RestoredTrueToyFluxMatrix.col(flux_it);
      FillHistFromEigenVector(cHist, RestoredTrueToyFluxVector);


      aHist->SetDirectory(td);
      aHist->Write();
      bHist->SetDirectory(td);
      bHist->Write();
      cHist->SetDirectory(td);
      cHist->Write();

    }

  }*/

  void Write1D(TDirectory *td, int nEbins) {

    for (int flux_it = 0; flux_it < TrueFluxMatrix.cols(); flux_it++) {

      TString truefh = TString::Format("TrueFlux_%i", flux_it+1);
      TH1D *aHist = new TH1D (truefh, truefh, nEbins, 0.0, 10.0);
      Eigen::VectorXd TrueFluxVector = TrueFluxMatrix.col(flux_it);
      FillHistFromEigenVector(aHist, TrueFluxVector);

      TString recofh = TString::Format("RecoFlux_%i", flux_it+1);
      TH1 *bHist = static_cast<TH1 *>(aHist->Clone(recofh));
      bHist->Reset();
      // TH1D *bHist = new TH1D (recofh, recofh, nEbins, 0.0, 10.0);
      Eigen::VectorXd RecoFluxVector = RecoFluxMatrix.col(flux_it);
      FillHistFromEigenVector(bHist, RecoFluxVector);

      TString restfh = TString::Format("RestoredTrueFlux_%i", flux_it+1);
      TH1 *cHist = static_cast<TH1 *>(aHist->Clone(restfh));
      cHist->Reset();
      Eigen::VectorXd RestoredTrueFluxVector = RestoredTrueFluxMatrix.col(flux_it);
      FillHistFromEigenVector(cHist, RestoredTrueFluxVector);


      aHist->SetDirectory(td);
      aHist->Write();
      bHist->SetDirectory(td);
      bHist->Write();
      cHist->SetDirectory(td);
      cHist->Write();

    }

  }

};

#ifdef ERS_WRAP_IN_NAMESPACE
}
#endif

#endif
