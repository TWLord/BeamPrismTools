#ifndef ETRUEERECO_HXX_SEEN
#define ETRUEERECO_HXX_SEEN

#include "BargerPropagator.h"

#include "Eigen/Dense"
#include "Eigen/StdVector"
#include "Eigen/Sparse"
// #include "Eigen/SparseCore"
#include "Eigen/IterativeLinearSolvers"

#include "TFile.h"
#include "TGraph.h"
#include "TGraph2D.h"
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

double deriv(double *evals, double step) {
  return (-evals[2] + 8 * evals[1] - 8 * evals[-1] + evals[-2]) / (12 * step);
}

double second_deriv(double *evals, double step) {
  double first_deriv_evals[5];

  first_deriv_evals[0] = deriv(&evals[-2], step);
  first_deriv_evals[1] = deriv(&evals[-1], step);
  first_deriv_evals[3] = deriv(&evals[1], step);
  first_deriv_evals[4] = deriv(&evals[2], step);

  return deriv(&first_deriv_evals[2], step);
}

TGraph2D *MatrixToTGraph(Eigen::MatrixXd aMat) {
  TGraph2D * graph1 = new TGraph2D(aMat.rows()*aMat.cols());
  int counter = 0;
  double stepx = 1.0/aMat.cols();
  double stepy = 1.0/aMat.rows();
  // offset = 1/2 stepx
  for (int row_it = 0; row_it < aMat.rows(); row_it++) {
    for (int col_it = 0; col_it < aMat.cols(); col_it++) {
      // graph1->SetPoint( counter, col_it*11/aMat.cols(), row_it*11/aMat.rows(), aMat(row_it, col_it) );
      // graph1->SetPoint( counter, float(col_it)*10/aMat.cols(), float(row_it)*10/aMat.rows(), aMat(row_it, col_it) );
      graph1->SetPoint( counter, (col_it+0.5)*stepx, (row_it+0.5)*stepy, aMat(row_it, col_it) );
      counter++;
    }
  }
  return graph1;
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
    enum ToyMatrixType { mRandom, mRandomLimited, mGauss, mEGauss }; 
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

    // p.algo_id = Params::kSVD;
    // p.algo_id = Params::kConjugateGradient;
    p.algo_id = Params::kInverse;
    // p.toyM_id = Params::mRandom;
    p.toyM_id = Params::mGauss;
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
  	      // int nXbins = tmpFlux2D->GetXaxis()->GetNbins();
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

  Eigen::MatrixXd ROOTMatrixRebin(Eigen::MatrixXd aMat, int newrows, int newcols, bool rescale = true ) {

    TH2D *aHist = new TH2D ("hname", "hname", aMat.rows(), 0.0, 10.0, aMat.cols(), 0.0, 10.0);

    for (int row_it = 0; row_it < aMat.rows(); row_it++) {
      for (int col_it = 0; col_it < aMat.cols(); col_it++) {
        aHist->SetBinContent( row_it+1, col_it+1, aMat(row_it, col_it));
      }
    }

    int MergeX = aMat.rows()/newrows;
    int MergeY = aMat.cols()/newcols;
    aHist->Rebin2D(MergeX, MergeY);
    if (rescale) {
      aHist->Scale(1.0 / double(MergeX * MergeY));
    }

    // If binning not divisible then fills a larger matrix with extra rows
    /*Eigen::MatrixXd newMat = Eigen::MatrixXd::Zero(aHist->GetXaxis()->GetNbins(), aHist->GetYaxis()->GetNbins());
    for (int row_it = 0; row_it < newMat.rows(); row_it++) {
      for (int col_it = 0; col_it < newMat.cols(); col_it++) {
        newMat(row_it, col_it) = aHist->GetBinContent( row_it+1, col_it+1 );
      }
    }*/

    // If binning not divisible then fills to specified new matrix size but removes off last row(s)/col(s)
    Eigen::MatrixXd newMat = Eigen::MatrixXd::Zero(newrows, newcols);
    for (int row_it = 0; row_it < newMat.rows(); row_it++) {
      for (int col_it = 0; col_it < newMat.cols(); col_it++) {
        newMat(row_it, col_it) = aHist->GetBinContent( row_it+1, col_it+1 );
      }
    }

    return newMat;
  }

  Eigen::MatrixXd MatrixRebinRows(Eigen::MatrixXd aMat, int newrows, bool rescale = true ) {
    int oldrows = aMat.rows();
    int mergedrows = oldrows/newrows; 

    // std::cout << "oldrows : " << oldrows << std::endl;
    // std::cout << "newrows : " << newrows << std::endl;
    // std::cout << "mergedrows : " << mergedrows << std::endl;

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
	  // std::cout << "Entry for (" << row_it << "," << col_it << ") : " << newMat(row_it, col_it) << std::endl;
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

  void testLowDimFit(int Ebins, int NFluxes, double SenseSmearingLimit, double SSLpb, double NoiseSmearingLimit, bool SmearSensingMatrix, bool SmearRecoFlux, std::string OutputFile, int lessEbins) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << "Running lower dimension fit test with toy fluxes" << std::endl;

    int inputEbins = Ebins;
    if (Ebins != 400/(400/Ebins)) {
      std::cout << "Changed binning for Ebins from " << Ebins << " to " <<  400/(400/Ebins) << std::endl;
      Ebins = 400/(400/Ebins);
    }

    // Initialising here with Ebins as higher dim binning, lessEbins lower dim binning
    LoadToySensingMatrix(Ebins, SenseSmearingLimit, SSLpb);

    RecoFluxMatrix = TrueSensingMatrix * TrueFluxMatrix; 

    // Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(Ebins, Ebins*2); 
    Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(Ebins, Ebins); 
    Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix; 

    Eigen::MatrixXd ldTrueToyFluxMatrix = MatrixRebinRows(TrueToyFluxMatrix, lessEbins);
    Eigen::MatrixXd ldRecoToyFluxMatrix = MatrixRebinRows(RecoToyFluxMatrix, lessEbins);

    // maybe consider testing MatrixRebinRows eg:
    std::cout << "\nToy Fluxes Binned to " << lessEbins << " rows : \n"<< std::endl;
    std::cout << ldRecoToyFluxMatrix.format(CleanFmt) << std::endl;

    // Eigen::MatrixXd lldRecoToyFluxMatrix = MatrixRebinRows(ldRecoToyFluxMatrix, lessEbins/2);
    // std::cout << "\nRebinned to " << lessEbins/2 << " rows : \n"<< std::endl;
    // Eigen::MatrixXd lldRecoToyFluxMatrix = MatrixRebinRows(ldRecoToyFluxMatrix, 4);
    // std::cout << "\nRebinned to " << 4 << " rows : \n"<< std::endl;

    // std::cout << lldRecoToyFluxMatrix.format(CleanFmt) << std::endl;

    // Eigen::MatrixXd ROOTRebin = ROOTMatrixRebin(ldRecoToyFluxMatrix, lessEbins/2, ldRecoToyFluxMatrix.cols());
    // std::cout << "\nROOT Rebin to " << lessEbins/2 << " rows : \n"<< std::endl;
    // Eigen::MatrixXd ROOTRebin = ROOTMatrixRebin(ldRecoToyFluxMatrix, 4, ldRecoToyFluxMatrix.cols());
    // std::cout << "\nROOT Rebin to " << 4 << " rows : \n"<< std::endl;

    // std::cout << ROOTRebin.format(CleanFmt) << std::endl;

    ComputeMatrix(MatrixRebinCols(ldTrueToyFluxMatrix, lessEbins), MatrixRebinCols(ldRecoToyFluxMatrix, lessEbins));
    Eigen::MatrixXd fittedSensingMatrix = RecoSensingMatrix;
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix( ldTrueToyFluxMatrix, ldRecoToyFluxMatrix, true);

    TGraph2D *fittedSensingGraph = MatrixToTGraph(fittedSensingMatrix);
    fittedSensingGraph->SetName("fittedSensingGraph");

    /*std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
    std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
    std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
    std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;*/

    Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, Ebins); 
    // Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, largerEbins); 
    // std::cout << "Scaled up one" << std::endl;
    Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), Ebins); 
    // Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), largerEbins); 
    // std::cout << "Scaled up two" << std::endl;
    Eigen::MatrixXd InterpolatedMat = BiLinearInterpolate(fittedSensingGraph, Ebins);
    Eigen::MatrixXd InterpolatedMatInv = InterpolatedMat.inverse();
    // may need to re-normalise along cols after this

    // std::cout << "ScaledUpFittedSensingMatrix.rows() : " << ScaledUpFittedSensingMatrix.rows() << std::endl;
    // std::cout << "ScaledUpFittedSensingMatrix.cols() : " << ScaledUpFittedSensingMatrix.cols() << std::endl;

    TFile *f = CheckOpenFile(OutputFile, "RECREATE");

    if ( FDFluxVector.size() ) {
      // Fixing naming conventions
      Eigen::VectorXd FDFluxVectorHD = FDFluxVector;
      FDFluxVector = VectorRebin(FDFluxVectorHD, Ebins, true);
      // FDFluxVector = FDFluxVectorRebinned;
      Eigen::VectorXd FDFluxVectorRebinned = VectorRebin(FDFluxVector, lessEbins, true); 
      // Getting Recos
      Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVector;
      Eigen::VectorXd FDRecoVectorRebinned = VectorRebin(FDRecoVector, lessEbins, true); 
      Eigen::VectorXd FDRestoredVectorRebinned = fittedSensingMatrix.inverse() * FDRecoVectorRebinned; 
      std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
      Eigen::VectorXd FDRestoredVectorLowRank = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);
      std::cout << " Applying scaled-up sensing matrix to FD Reco Vector " << std::endl;
      Eigen::VectorXd FDRestoredVectorScaled = ScaledUpFittedSensingMatrix.inverse()*FDRecoVector;
      Eigen::VectorXd FDRestoredVectorScaledInv = ScaledUpFittedSensingMatrixInv*FDRecoVector;

      Eigen::VectorXd FDRestoredVectorBiLinear = InterpolatedMat.inverse()*FDRecoVector;

      WriteVector(f, FDFluxVectorHD.size(), FDFluxVectorHD, "FDTrueVectorHD"); 
      WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector");
      WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector");
      WriteVector(f, FDRestoredVectorLowRank.size(), FDRestoredVectorLowRank, "FDRestoredVectorLowRank");
      WriteVector(f, FDRestoredVectorScaled.size(), FDRestoredVectorScaled, "FDRestoredVectorScaled");
      WriteVector(f, FDRestoredVectorScaledInv.size(), FDRestoredVectorScaledInv, "FDRestoredVectorScaledInv");
      WriteVector(f, FDFluxVectorRebinned.size(), FDFluxVectorRebinned, "FDFluxVectorRebinned" );
      WriteVector(f, FDRecoVectorRebinned.size(), FDRecoVectorRebinned, "FDRecoVectorRebinned" );
      WriteVector(f, FDRestoredVectorRebinned.size(), FDRestoredVectorRebinned, "FDRestoredVectorRebinned" );

      /*WriteVector(f, FDFluxVectorHD.size(), FDFluxVectorHD, "FDTrueVectorHD", FDFluxOriginal.get());
      WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
      WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorLowRank.size(), FDRestoredVectorLowRank, "FDRestoredVectorLowRank", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaled.size(), FDRestoredVectorScaled, "FDRestoredVectorScaled", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaledInv.size(), FDRestoredVectorScaledInv, "FDRestoredVectorScaledInv", FDFluxOriginal.get());
      WriteVector(f, FDFluxVectorRebinned.size(), FDFluxVectorRebinned, "FDFluxVectorRebinned" );
      WriteVector(f, FDRecoVectorRebinned.size(), FDRecoVectorRebinned, "FDRecoVectorRebinned" );
      WriteVector(f, FDRestoredVectorRebinned.size(), FDRestoredVectorRebinned, "FDRestoredVectorRebinned" );*/

      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix, "TrueSensingMatrix");  
      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix.inverse(), "TrueSensingMatrixInv");  
      WriteMatrix2D(f, fittedSensingMatrix.cols(), fittedSensingMatrix.rows(), fittedSensingMatrix, "fittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrix.cols(), ScaledUpFittedSensingMatrix.rows(), ScaledUpFittedSensingMatrix, "ScaledUpFittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrixInv.cols(), ScaledUpFittedSensingMatrixInv.rows(), ScaledUpFittedSensingMatrixInv, "ScaledUpFittedSensingMatrixInv");  
      WriteMatrix2D(f, InterpolatedMat.cols(), InterpolatedMat.rows(), InterpolatedMat, "InterpolatedMat");  
      WriteMatrix2D(f, InterpolatedMatInv.cols(), InterpolatedMatInv.rows(), InterpolatedMatInv, "InterpolatedMatInv");  


      Eigen::MatrixXd TrueSMrebin = MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, lessEbins, false), lessEbins, true);
      WriteMatrix2D(f, TrueSMrebin.cols(), TrueSMrebin.rows(), TrueSMrebin, "TrueSMrebin");  

      FDFluxOriginal->SetName("FDFluxOriginal");
      FDFluxOriginal->Write();

      fittedSensingGraph->Write();
    }

    f->Close();

  }

  void doMatrixFitAnalysis(int Ebins, int NFluxes, double SenseSmearingLimit, double SSLpb, double NoiseSmearingLimit, bool SmearSensingMatrix, bool SmearRecoFlux, std::string OutputFile) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    Ebins = TrueFluxMatrix.rows();
    std::cout << "TrueFluxMatrix.rows() = " << Ebins << std::endl;
    std::cout << "FluxMatrix_Full.rows() = " << FluxMatrix_Full.rows() << std::endl;

    LoadToySensingMatrix(FluxMatrix_Full.rows(), SenseSmearingLimit, SSLpb);
    RecoFluxMatrix = TrueSensingMatrix * FluxMatrix_Full; 

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
    std::cout << "RebinnedSmearedRecoFluxMatrix.cols() : " << RebinnedSmearedRecoFluxMatrix.cols() << std::endl;

    // Fit sensing matrix with TrueFluxMatrix : rebin rows to get Ebins as required 
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix( MatrixRebinRows(TrueFluxMatrix, Ebins), RebinnedSmearedRecoFluxMatrix, true);
    // This one works well but takes awhile to run

    // Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(FluxMatrix_Full.rows(), NFluxes);
    Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(FluxMatrix_Full.rows(), FluxMatrix_Full.rows());
    std::cout << "NFluxes : " << NFluxes << std::endl;
    std::cout << "FluxMatrix_Full.rows() : " << FluxMatrix_Full.rows() << std::endl;
    Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix;
    Eigen::MatrixXd SmearedRecoToyFluxMatrix = SmearMatrix(RecoToyFluxMatrix, NoiseSmearingLimit);
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueToyFluxMatrix, RecoToyFluxMatrix);
    Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix( MatrixRebinRows(TrueToyFluxMatrix, Ebins), MatrixRebinRows(SmearedRecoToyFluxMatrix, Ebins), true);


    std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
    std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
    std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
    std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
    Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, FluxMatrix_Full.rows());
    Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), FluxMatrix_Full.rows());

    TFile *f = CheckOpenFile(OutputFile, "RECREATE");

    if ( FDFluxVector.size() ) {
      Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVector;
      Eigen::VectorXd FDRecoVectorRebinned = VectorRebin(FDRecoVector, Ebins, true); 
      Eigen::VectorXd FDRestoredVectorRebinned = fittedSensingMatrix.inverse() * FDRecoVectorRebinned; 
      std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
      Eigen::VectorXd FDRestoredVectorLowRank = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);
      std::cout << " Applying scaled-up sensing matrix to FD Reco Vector " << std::endl;
      Eigen::VectorXd FDRestoredVectorScaled = ScaledUpFittedSensingMatrix.inverse()*FDRecoVector;
      Eigen::VectorXd FDRestoredVectorScaledInv = ScaledUpFittedSensingMatrixInv*FDRecoVector;

      WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
      WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorLowRank.size(), FDRestoredVectorLowRank, "FDRestoredVectorLowRank", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaled.size(), FDRestoredVectorScaled, "FDRestoredVectorScaled", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaledInv.size(), FDRestoredVectorScaledInv, "FDRestoredVectorScaledInv", FDFluxOriginal.get());
      WriteVector(f, FDFluxVectorRebinned.size(), FDFluxVectorRebinned, "FDFluxVectorRebinned" );
      WriteVector(f, FDRecoVectorRebinned.size(), FDRecoVectorRebinned, "FDRecoVectorRebinned" );
      WriteVector(f, FDRestoredVectorRebinned.size(), FDRestoredVectorRebinned, "FDRestoredVectorRebinned" );

      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix, "TrueSensingMatrix");  
      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix.inverse(), "TrueSensingMatrixInv");  
      WriteMatrix2D(f, fittedSensingMatrix.cols(), fittedSensingMatrix.rows(), fittedSensingMatrix, "fittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrix.cols(), ScaledUpFittedSensingMatrix.rows(), ScaledUpFittedSensingMatrix, "ScaledUpFittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrixInv.cols(), ScaledUpFittedSensingMatrixInv.rows(), ScaledUpFittedSensingMatrixInv, "ScaledUpFittedSensingMatrixInv");  


      Eigen::MatrixXd TrueSMrebin = MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true);
      WriteMatrix2D(f, TrueSMrebin.cols(), TrueSMrebin.rows(), TrueSMrebin, "TrueSMrebin");  

      FDFluxOriginal->SetName("FDFluxOriginal");
      FDFluxOriginal->Write();
    }

    f->Close();

  }

  void doMatrixMapAnalysis(int Ebins, int NFluxes, double SenseSmearingLimit, double SSLpb, double NoiseSmearingLimit, bool SmearSensingMatrix, bool SmearRecoFlux, std::string OutputFile, double reg_param) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    Ebins = TrueFluxMatrix.rows();
    std::cout << "TrueFluxMatrix.rows() = " << Ebins << std::endl;
    std::cout << "FluxMatrix_Full.rows() = " << FluxMatrix_Full.rows() << std::endl;

    bool debug = false;
    if (debug) {
      // Do solve without smearing reco matrices - for testing
      LoadToySensingMatrix(Ebins, SenseSmearingLimit, SSLpb);
      Eigen::MatrixXd RebinnedTrueFluxMatrix = MatrixRebinRows(FluxMatrix_Full, Ebins);
      std::cout << " --- Rebin 1 : True Fluxes --- " << std::endl;
      // std::cout << RebinnedTrueFluxMatrix.format(CleanFmt) << std::endl; 
      Eigen::MatrixXd aSquareTrueFluxMatrix = MatrixRebinCols(RebinnedTrueFluxMatrix, Ebins);
      std::cout << " --- Rebin 2 : True Fluxes --- " << std::endl;
      Eigen::MatrixXd SquareSmearedRecoFluxMatrix = TrueSensingMatrix * aSquareTrueFluxMatrix;
      // reg_param = 0.5;
      double res_norm = 0;
      double soln_norm = 0;
      std::cout << "TrueFluxMatrix.rows() : " << TrueFluxMatrix.rows() << std::endl;
      std::cout << "Ebins : " << Ebins << std::endl;
      Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( aSquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix, reg_param, res_norm, soln_norm);

      std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
      std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
      // Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), Ebins*5);
      Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, FluxMatrix_Full.rows());
      Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), FluxMatrix_Full.rows());
      std::cout << " --- Scaled Up Fitted Sensing Matrix Inverse --- " << std::endl;
      // std::cout << ScaledUpFittedSensingMatrixInv.format(CleanFmt) << std::endl;

      std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
      std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
      std::cout << "Frob Norm of (Mat * Etrue) - Ereco : " << FrobeniusNorm((fittedSensingMatrix * aSquareTrueFluxMatrix) - SquareSmearedRecoFluxMatrix) << std::endl;

      /*Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(Ebins, Ebins);
      Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix;
      fittedSensingMatrix = SolveSensingMatrix( TrueToyFluxMatrix, RecoToyFluxMatrix, reg_param, res_norm, soln_norm);
      std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
      std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
      std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
      std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
      std::cout << "Frob Norm of (Mat * Etrue) - Ereco : " << FrobeniusNorm((fittedSensingMatrix * TrueToyFluxMatrix) - RecoToyFluxMatrix) << std::endl;*/

      TFile *f = CheckOpenFile(OutputFile, "RECREATE");
      if ( FDFluxVector.size() ) {
        Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVectorRebinned;
        std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
        Eigen::VectorXd FDRestoredVectorLowRank = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);
        std::cout << " Applying scaled-up sensing matrix to FD Reco Vector " << std::endl;
        std::cout << " ScaledUpFittedSensingMatrix.cols() : " << ScaledUpFittedSensingMatrix.cols() << std::endl;
        std::cout << " ScaledUpFittedSensingMatrix.rows() : " << ScaledUpFittedSensingMatrix.rows() << std::endl;
        std::cout << " FDRecoVector.rows() : " << FDRecoVector.rows() << std::endl;
        Eigen::VectorXd FDRestoredVectorScaled = ScaledUpFittedSensingMatrix.inverse()*FDRecoVector;
        Eigen::VectorXd FDRestoredVectorScaledInv = ScaledUpFittedSensingMatrixInv*FDRecoVector;

        WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
        WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
        WriteVector(f, FDRestoredVectorLowRank.size(), FDRestoredVectorLowRank, "FDRestoredVectorLowRank", FDFluxOriginal.get());
        WriteVector(f, FDRestoredVectorScaled.size(), FDRestoredVectorScaled, "FDRestoredVectorScaled", FDFluxOriginal.get());
        WriteVector(f, FDRestoredVectorScaledInv.size(), FDRestoredVectorScaledInv, "FDRestoredVectorScaledInv", FDFluxOriginal.get());

        WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix, "TrueSensingMatrix");  
        WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix.inverse(), "TrueSensingMatrixInv");  
        WriteMatrix2D(f, fittedSensingMatrix.cols(), fittedSensingMatrix.rows(), fittedSensingMatrix, "fittedSensingMatrix");  
        WriteMatrix2D(f, ScaledUpFittedSensingMatrix.cols(), ScaledUpFittedSensingMatrix.rows(), ScaledUpFittedSensingMatrix, "ScaledUpFittedSensingMatrix");  
        WriteMatrix2D(f, ScaledUpFittedSensingMatrixInv.cols(), ScaledUpFittedSensingMatrixInv.rows(), ScaledUpFittedSensingMatrixInv, "ScaledUpFittedSensingMatrixInv");  

        FDFluxOriginal->SetName("FDFluxOriginal");
        FDFluxOriginal->Write();
      }

      f->Close();
      std::cin.ignore();
      return;
    }

    LoadToySensingMatrix(FluxMatrix_Full.rows(), SenseSmearingLimit, SSLpb);
    RecoFluxMatrix = TrueSensingMatrix * FluxMatrix_Full; 

    if (SmearRecoFlux) {
      std::cout << "Smear Reco Fluxes" << std::endl;
      SmearedRecoFluxMatrix = SmearMatrix(RecoFluxMatrix, NoiseSmearingLimit);
      std::cout << " --- Smeared Reco Fluxes --- " << std::endl;
      // std::cout << SmearedRecoFluxMatrix.format(CleanFmt) << std::endl;
    } else {
      SmearedRecoFluxMatrix = RecoFluxMatrix;
    }

    Eigen::MatrixXd RebinnedSmearedRecoFluxMatrix = MatrixRebinRows(SmearedRecoFluxMatrix, Ebins);
    std::cout << " --- Rebin 1 : Reco Fluxes --- " << std::endl;
    // std::cout << RebinnedSmearedRecoFluxMatrix.format(CleanFmt) << std::endl; 
    Eigen::MatrixXd SquareSmearedRecoFluxMatrix = MatrixRebinCols(RebinnedSmearedRecoFluxMatrix, Ebins);
    std::cout << " --- Rebin 2 : Reco Fluxes --- " << std::endl;
    // std::cout << SquareSmearedRecoFluxMatrix.format(CleanFmt) << std::endl; 

    // ComputeMatrix(SquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix);
    // std::cout << "RebinnedSmearedRecoFluxMatrix.cols() : " << RebinnedSmearedRecoFluxMatrix.cols() << std::endl;

    Eigen::MatrixXd RebinnedTrueFluxMatrix = MatrixRebinRows(FluxMatrix_Full, Ebins);
    std::cout << " --- Rebin 1 : True Fluxes --- " << std::endl;
    // std::cout << RebinnedTrueFluxMatrix.format(CleanFmt) << std::endl; 
    Eigen::MatrixXd aSquareTrueFluxMatrix = MatrixRebinCols(RebinnedTrueFluxMatrix, Ebins);
    std::cout << " --- Rebin 2 : True Fluxes --- " << std::endl;


    // Set negative values to positive small values & fit with remaining dataset 

    // Fit sensing matrix with TrueFluxMatrix - set to square starting mats
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(SquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix);

    // Fit sensing matrix with TrueFluxMatrix : rebin rows to get Ebins as required 
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix( MatrixRebinRows(TrueFluxMatrix, Ebins), RebinnedSmearedRecoFluxMatrix, true); // This one works well but takes awhile to run

    // Fit sensing matrix with FluxMatrix_Full - all fluxes
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(FluxMatrix_Full, (TrueSensingMatrix*FluxMatrix_Full) ); // untested - binning probably wrong

    // reg_param = 0.001;
    double res_norm = 0;
    double soln_norm = 0;
    std::cout << "TrueFluxMatrix.rows() : " << TrueFluxMatrix.rows() << std::endl;
    std::cout << "Ebins : " << Ebins << std::endl;
    // Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( RebinnedTrueFluxMatrix, RebinnedSmearedRecoFluxMatrix, reg_param, res_norm, soln_norm);
    
    bool iterativesolve = false; // should replace with fParams.iterativesolve..
    if (iterativesolve) {

      int nsteps = 30;
      double CSNorm = -3;
      double StabilityFactor=-3;
      // looking at coeff removal below coefflim size
      // TGraph coeffs(nsteps);
      int ncoeffs = pow(Ebins, 2);
      std::cout << "ncoeffs : " << ncoeffs << std::endl;
      TH2D *CoeffChange =
         new TH2D("CoeffChange2D", "Coeff Change with steps", ncoeffs, 0, ncoeffs, nsteps+1, 0, nsteps+1);
      TH2D *WeightChange =
         new TH2D("WeightChange2D", "Weighting Change with steps", ncoeffs, 0, ncoeffs, nsteps+1, 0, nsteps+1);
  
      double reg_exp = 1E-5;
      std::vector<double> omega;

      Eigen::MatrixXd fittedSensingMatrix;
  
      std::vector<double> eta_hat, rho_hat;
      for (size_t l_it = 0; l_it < nsteps; ++l_it) {
  
        fittedSensingMatrix = IterativeSolveSensingMatrix( aSquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix, pow(10, reg_exp), res_norm, soln_norm, omega);
        std::cout << "soln_norm : " << soln_norm << std::endl;
        std::cout << "res_norm : " << res_norm << std::endl;
        eta_hat.push_back(log(soln_norm));
        rho_hat.push_back(log(res_norm));
  
  
        // std::cout << "\n --------------- Solve Coeffs --------------- " << std::endl;
        double largecoeffs = 0;
        double coeffsum = 0;

        for (size_t i = 0; i < omega.size(); i++) {
          // std::cout << omega[i] << std::endl;
          CoeffChange->SetBinContent( i+1, l_it+1, omega[i]);
        }
  
        int w_it = 0;
        if ( l_it != (nsteps - 1) ) { // Skip last reweight to do filtering
          for (double &weight : omega) {
            weight = std::abs( pow(10, CSNorm) /((pow(10,-CSNorm)*weight) + pow(10,StabilityFactor)) );
            // weight = std::abs( pow(10, reg_exp) /((pow(10,-reg_exp)*weight) + pow(10,StabilityFactor)) );
            WeightChange->SetBinContent( w_it+1, l_it+1, weight);
            w_it++;
          }
        }
      }
  
      std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
      std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
      std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
      std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
      std::cout << "Frob Norm of (Mat * Etrue) - Ereco : " << FrobeniusNorm((fittedSensingMatrix * aSquareTrueFluxMatrix) - SquareSmearedRecoFluxMatrix) << std::endl;

      TFile *f = CheckOpenFile(OutputFile, "RECREATE");
      if ( FDFluxVector.size() ) {
        Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVector;
        std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
        Eigen::VectorXd FDRestoredVector = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);

        WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
        WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
        WriteVector(f, FDRestoredVector.size(), FDRestoredVector, "FDRestoredVector", FDFluxOriginal.get());

        FDFluxOriginal->SetName("FDFluxOriginal");
        FDFluxOriginal->Write();

        CoeffChange->Write();
        WeightChange->Write();
      }

      f->Close();
      std::cin.ignore();
      return;

    }

    // v lcurve solver
    bool curvesolve = false; // should replace with fParams..
    if (curvesolve) {
      double start = -6;
      double end = 1;
      int nsteps = 6000;
      TGraph lcurve(nsteps);
      TGraph kcurve(nsteps);
      double step = double(end - start) / double(nsteps);
      std::vector<double> eta_hat, rho_hat;
      for (size_t l_it = 0; l_it < nsteps; ++l_it) {
        std::cout << "\n\n ------ Step : " << l_it << " ------ " << std::endl;
        double reg_exp = start + double(l_it) * step;
        // Passed parameter is regularization factor, should scan for the best one,
        // double soln_norm, res_norm;
        // fls.Solve(pow(10, reg_exp), BCRegFactor, res_norm, soln_norm);
        Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( aSquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix, pow(10, reg_exp), res_norm, soln_norm);
        std::cout << "soln_norm : " << soln_norm << std::endl;
        std::cout << "res_norm : " << res_norm << std::endl;
        eta_hat.push_back(log(soln_norm));
        rho_hat.push_back(log(res_norm));
  
        lcurve.SetPoint(l_it, rho_hat.back() / 2.0, eta_hat.back() / 2.0);
      }
  
      double max_curv = -std::numeric_limits<double>::max();
      double best_reg;
      for (size_t l_it = 4; l_it < (nsteps - 4); ++l_it) {
        double curv =
            2.0 *
            (deriv(&rho_hat[l_it], step) * second_deriv(&eta_hat[l_it], step) -
             deriv(&eta_hat[l_it], step) * second_deriv(&rho_hat[l_it], step)) /
            pow(pow(deriv(&rho_hat[l_it], step), 2) +
                    pow(deriv(&eta_hat[l_it], step), 2),
                3 / 2);
        kcurve.SetPoint(l_it - 4, start + double(l_it) * step, curv);

        if (curv > max_curv) {
          max_curv = curv;
          best_reg = pow(10, start + double(l_it) * step);
        }
      }

      Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( aSquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix, best_reg, res_norm, soln_norm);

      std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
      std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
      std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
      std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
      std::cout << "Frob Norm of (Mat * Etrue) - Ereco : " << FrobeniusNorm((fittedSensingMatrix * aSquareTrueFluxMatrix) - SquareSmearedRecoFluxMatrix) << std::endl;

      TFile *f = CheckOpenFile(OutputFile, "RECREATE");
      if ( FDFluxVector.size() ) {
        Eigen::VectorXd FDRecoVector = TrueSensingMatrix*FDFluxVector;
        std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
        Eigen::VectorXd FDRestoredVector = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);

        WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
        WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
        WriteVector(f, FDRestoredVector.size(), FDRestoredVector, "FDRestoredVector", FDFluxOriginal.get());

        FDFluxOriginal->SetName("FDFluxOriginal");
        FDFluxOriginal->Write();

	lcurve.Write("lcurve");
	kcurve.Write("kcurve");
      }

      f->Close();
      std::cin.ignore();
      return;
    }

    // v normal solver
    Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( aSquareTrueFluxMatrix, SquareSmearedRecoFluxMatrix, reg_param, res_norm, soln_norm);

    std::cout << TrueSensingMatrix.cols() << "," << TrueSensingMatrix.rows() << std::endl;
    std::cout << FluxMatrix_Full.cols() << "," << FluxMatrix_Full.rows() << std::endl;
    /*
    Eigen::MatrixXd TrueToyFluxMatrix = LoadToyFluxes(FluxMatrix_Full.rows(), FluxMatrix_Full.cols());
    Eigen::MatrixXd RecoToyFluxMatrix = TrueSensingMatrix * TrueToyFluxMatrix;

    Eigen::MatrixXd SmearedRecoToyFluxMatrix = SmearMatrix(RecoToyFluxMatrix, NoiseSmearingLimit);
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueToyFluxMatrix, RecoToyFluxMatrix);
    Eigen::MatrixXd sqTrueMat = MatrixRebinCols( MatrixRebinRows(TrueToyFluxMatrix, Ebins), Ebins);
    Eigen::MatrixXd sqRecoMat = MatrixRebinCols( MatrixRebinRows(SmearedRecoToyFluxMatrix, Ebins), Ebins);
    std::cout << " -------- and ------- "<< std::endl;
    std::cout << sqTrueMat.format(CleanFmt) << std::endl;
    std::cout << " -------- and ------- "<< std::endl;
    std::cout << sqRecoMat.format(CleanFmt) << std::endl;
    Eigen::MatrixXd fittedSensingMatrix = SolveSensingMatrix( sqTrueMat, sqRecoMat, reg_param, res_norm, soln_norm);
    // Eigen::MatrixXd fittedSensingMatrix = fitSensingMatrix(TrueToyFluxMatrix, SmearedRecoToyFluxMatrix);
    // Eigen::MatrixXd fittedSensingMatrix = RecoSensingMatrix;
    */

    std::cout << " --- Fitted Sensing Matrix --- " << std::endl;
    std::cout << fittedSensingMatrix.format(CleanFmt) << std::endl;
    Eigen::MatrixXd ScaledUpFittedSensingMatrix = scaleUpSensingMatrix(fittedSensingMatrix, FluxMatrix_Full.rows());
    Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), FluxMatrix_Full.rows());
    std::cout << " --- Scaled Up Fitted Sensing Matrix Inverse --- " << std::endl;
    std::cout << " --- Rebinned True Sensing Matrix --- " << std::endl;
    std::cout << MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true).format(CleanFmt) << std::endl;
    std::cout << "Frob Norm of (Mat * Etrue) - Ereco : " << FrobeniusNorm((fittedSensingMatrix * aSquareTrueFluxMatrix) - SquareSmearedRecoFluxMatrix) << std::endl;
    //////////////// Eigen::MatrixXd ScaledUpFittedSensingMatrixInv = scaleUpSensingMatrix(fittedSensingMatrix.inverse(), FluxMatrix_Full.rows());
   
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
      std::cout << " Applying low-rank sensing matrix to FD Reco Vector " << std::endl;
      Eigen::VectorXd FDRestoredVectorLowRank = applyLowRankSensingMatrix( fittedSensingMatrix.inverse(), FDRecoVector);
      std::cout << " Applying scaled-up sensing matrix to FD Reco Vector " << std::endl;
      std::cout << " ScaledUpFittedSensingMatrix.cols() : " << ScaledUpFittedSensingMatrix.cols() << std::endl;
      std::cout << " ScaledUpFittedSensingMatrix.rows() : " << ScaledUpFittedSensingMatrix.rows() << std::endl;
      std::cout << " FDRecoVector.rows() : " << FDRecoVector.rows() << std::endl;
      Eigen::VectorXd FDRestoredVectorScaled = ScaledUpFittedSensingMatrix.inverse()*FDRecoVector;
      Eigen::VectorXd FDRestoredVectorScaledInv = ScaledUpFittedSensingMatrixInv*FDRecoVector;

      WriteVector(f, FDFluxVector.size(), FDFluxVector, "FDTrueVector", FDFluxOriginal.get());
      WriteVector(f, FDRecoVector.size(), FDRecoVector, "FDRecoVector", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorLowRank.size(), FDRestoredVectorLowRank, "FDRestoredVectorLowRank", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaled.size(), FDRestoredVectorScaled, "FDRestoredVectorScaled", FDFluxOriginal.get());
      WriteVector(f, FDRestoredVectorScaledInv.size(), FDRestoredVectorScaledInv, "FDRestoredVectorScaledInv", FDFluxOriginal.get());

      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix, "TrueSensingMatrix");  
      WriteMatrix2D(f, TrueSensingMatrix.cols(), TrueSensingMatrix.rows(), TrueSensingMatrix.inverse(), "TrueSensingMatrixInv");  
      WriteMatrix2D(f, fittedSensingMatrix.cols(), fittedSensingMatrix.rows(), fittedSensingMatrix, "fittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrix.cols(), ScaledUpFittedSensingMatrix.rows(), ScaledUpFittedSensingMatrix, "ScaledUpFittedSensingMatrix");  
      WriteMatrix2D(f, ScaledUpFittedSensingMatrixInv.cols(), ScaledUpFittedSensingMatrixInv.rows(), ScaledUpFittedSensingMatrixInv, "ScaledUpFittedSensingMatrixInv");  

      Eigen::MatrixXd TrueSMrebin = MatrixRebinRows( MatrixRebinCols(TrueSensingMatrix, Ebins, false), Ebins, true);
      WriteMatrix2D(f, TrueSMrebin.cols(), TrueSMrebin.rows(), TrueSMrebin, "TrueSMrebin");  

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
        for (int sub_it = col_it+aGevBins; sub_it >= 0; sub_it--) {
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
    } else if ( fParams.toyM_id == Params::mGauss ) {
      gRandom->SetSeed(0);
      for (int col_it = 0; col_it < nEbins; col_it++) {
        TString gstr = TString::Format("TMath::Gaus(x,0.5,%f)", lim/2);
        TF1* fgaus = new TF1("fgaus", gstr.Data(), 0.0, 1.0);
	float total = 0;
        // float limit = 0.2;
        float limit = lim;
        TrueSensingMatrix(col_it, col_it) = 1;
        total += 1;
	int aGevBins = (nEbins/10 + (nEbins%10 != 0))/10; // divide by X because want less than a gev..
        for (int sub_it = col_it+aGevBins; sub_it >= 0; sub_it--) {
          if ( sub_it == col_it || sub_it >= nEbins) {
	    continue;
	  }
	  float gausrandom = fgaus->Eval((limit/lim)*0.5);
	  limit -= gausrandom/50;
	  if ( limit <= 0 ) {
	    break;
          }
	  TrueSensingMatrix(sub_it, col_it) = gausrandom; 
	  total += gausrandom;
	}
        for (int row_it = 0; row_it < nEbins; row_it++) {
          TrueSensingMatrix(row_it, col_it) /= total;
        }
      }
    } else if ( fParams.toyM_id == Params::mEGauss ) {
      TH1* tmph = FDFluxOriginal->Rebin(FDFluxOriginal->GetNbinsX()/nEbins);
      tmph->Scale(1.0/nEbins);
      for (int col_it = 0; col_it < nEbins; col_it++) {
        // double Energy = tmph->GetBinCenter(col_it+1)*1000; // energy resolution defined in MeV
        double Energy = tmph->GetBinCenter(col_it+1); // energy resolution defined in MeV
        // std::cout << Energy << std::endl; 
        double sigma = Energy*0.15/sqrt(Energy);
        TString gstr = TString::Format("TMath::Gaus(x,%f,%f)", Energy, sigma);
        // TF1* fgaus = new TF1("fgaus", gstr.Data(), 0.0, 10000.0);
        TF1* fgaus = new TF1("fgaus", gstr.Data(), 0.0, 10.0);
	float total = 0;
        for (int row_it = 0; row_it < nEbins; row_it++) {
          // float fg = fgaus->Eval(tmph->GetBinCenter(row_it+1)*1000);
          float fg = fgaus->Eval(tmph->GetBinCenter(row_it+1));
          // std::cout << "fg : " << fg << std::endl;
	  TrueSensingMatrix(row_it, col_it) = fg; 
          total += fg;
        }
	// std::cout << "Normalising row /= " << total << std::endl;
        for (int row_it = 0; row_it < nEbins; row_it++) {
          TrueSensingMatrix(row_it, col_it) /= total;
        }
      }
    }

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << " --- True Sensing Matrix --- " << std::endl; 
    // std::cout << TrueSensingMatrix.format(CleanFmt) << std::endl; 
  }

  Eigen::MatrixXd LoadToyFluxes(int nEbins, int NFluxes) {
    std::cout << "Load Toy Fluxes" << std::endl;
    Eigen::MatrixXd ToyFluxes = Eigen::MatrixXd::Zero(nEbins, NFluxes);

    if (fParams.toyF_id == Params::fRandom) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int row_it = 0; row_it < nEbins; row_it++) {
        for (int col_it = 0; col_it < NFluxes; col_it++) {
	  float randomf = (float) rand()/RAND_MAX; 
          ToyFluxes(row_it, col_it) = randomf; 
	}
      }
    }

    if (fParams.toyF_id == Params::fRandomLimited) {
      srand( time(NULL) ); // Randomize seed initialization 
      for (int col_it = 0; col_it < NFluxes; col_it++) {
	if (NFluxes!=nEbins) {
          std::cout << "[ERROR]: Only valid for sq toy flux matrices" << std::endl;
	}
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
    // std::cout << ToyFluxes.format(CleanFmt) << std::endl; 

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

    Eigen::VectorXd RestoredVec = Eigen::VectorXd::Zero(RecoVec.size());

    std::cout << "RecoVec.size() : " << RecoVec.size() << std::endl;
    std::cout << "aMat.cols() : " << aMat.cols() << std::endl;
    // if ( ! RecoVec.size()%aMat.cols()) {
      int multiplier = RecoVec.size()/aMat.cols(); 

      for (int it = 0; it < RecoVec.size(); it++) {
        double newbin = 0;
        for (int sub_it = 0; sub_it < RecoVec.size(); sub_it++) {
          // catch it/multiplier and sub_it/multiplier >= aMat.rows()/cols() here
          // uses previous SM value if last one unavailable
          int roundit = it/multiplier; 
          if (it/multiplier >= aMat.rows()) {
            roundit = aMat.cols() - 1;
	    // std::cout << "roundit : " << roundit << std::endl;
          }
          int roundsubit = (sub_it/multiplier);
          if (sub_it/multiplier >= aMat.rows()) {
            roundsubit = aMat.cols() - 1; 
	    // std::cout << "roundsubit : " << roundsubit << std::endl;
	    // std::cout << "sub_it/multiplier : " << sub_it/multiplier << std::endl;
          }

	  if (sub_it/multiplier != it/multiplier) {
	    // std::cout << "using roundit : " << roundit << std::endl;
	    // std::cout << "    and using using roundsubit : " << roundsubit << std::endl;
            newbin += RecoVec(sub_it)*aMat(roundit, roundsubit)/multiplier;
	  }
	  if (sub_it/multiplier == it/multiplier) {
	    // std::cout << "using roundit only : " << roundit << std::endl;
            newbin += RecoVec(it)*aMat(roundit, roundit)/multiplier;
	  }
        }
        RestoredVec(it) = newbin; 
      }
    // } else { 
      // define some averaging between bins over non-divisible binning schema
      
    // }
    return RestoredVec;
  }

  Eigen::MatrixXd BiLinearInterpolate(TGraph2D * thegraph, int newbins) {
    Eigen::MatrixXd newMat(newbins, newbins);
    for (int row_it = 0; row_it < newMat.rows(); row_it++) {
      for (int col_it = 0; col_it < newMat.cols(); col_it++) {
        // double newx = 10*float(row_it)/newbins;
        // double newy = 10*float(col_it)/newbins;
        double newx = float(row_it)/newbins;
        double newy = float(col_it)/newbins;
        // + offset?
        if (!newx && !newy) {
          continue;
        }
        std::cout << "newx,newy : " << newx << "," << newy << std::endl;
        newMat(col_it, row_it) = thegraph->Interpolate(newx, newy);
      }
    }
    return newMat;
  }

  Eigen::MatrixXd scaleUpSensingMatrix(Eigen::MatrixXd aMat, int newbins) {
    // want to scale down splittings across rows, but preserve size across columns
    Eigen::MatrixXd newMat1 = Eigen::MatrixXd::Zero(newbins, aMat.cols());
    Eigen::MatrixXd newMat2 = Eigen::MatrixXd::Zero(newbins, newbins);

    Eigen::MatrixXd newMat3 = Eigen::MatrixXd::Zero(aMat.rows(), newbins);

    int multiplier = newbins/aMat.rows() + (newbins%aMat.rows() != 0);

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << " --- Matrix 0 --- " << std::endl;
    // std::cout << aMat.format(CleanFmt) << std::endl;


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

    // manually average matrix over adjacent rows
    float midbin = (multiplier/2.0) + 0.5;// 0.5 is "bin" center off-set
    // std::cout << "midbin : " << midbin << std::endl;
    if ( std::ceil(midbin) == std::floor(midbin) ) { // midbin center is NOT between bins
      // std::cout << "midbin center is NOT between bins " << std::endl;
      for (int row_it = 0; row_it < newMat1.rows(); row_it++) {
        for (int col_it = 0; col_it < newMat1.cols(); col_it++) {
	  double val;
          double Oldrow_it = row_it/multiplier;
          double Oldcol_it = col_it;
	  // check position wrt mid-bin
	  if ( ((row_it)%multiplier)+1 < midbin ) {
	    // std::cout << "less than - row_it+1 : " << row_it+1 << std::endl;
	    // check if you can get N-1 bincontents
	    if ( (row_it)/multiplier != 0) {
            // if possible, get and take weighted average
              double rowfrac = std::abs(((row_it%multiplier)-(midbin-1))/(multiplier));
	      // std::cout << "rowfrac : " << rowfrac << std::endl;
              val = (aMat(Oldrow_it-1, Oldcol_it) * (rowfrac)) + (aMat(Oldrow_it, Oldcol_it) * (1-rowfrac));
	    } else {
              val = aMat(Oldrow_it, Oldcol_it);
	    }
	  // check position wrt mid-bin
	  } else if ( ((row_it+1)%multiplier) == midbin ) {
	    // std::cout << "equals - row_it+1 : " << row_it+1 << std::endl;
	    val = aMat(Oldrow_it, Oldcol_it);
	  // check position wrt mid-bin
	  } else if ( ((row_it)%multiplier)+1 > midbin ) {
	    // std::cout << "more than - row_it+1 : " << row_it+1 << std::endl;
	    // check if you can get N+1 bincontents
	    if ( (row_it)/multiplier != (aMat.rows()-1) ) {
            // if possible, get and take weighted average
              double rowfrac = std::abs(((row_it%multiplier)-(midbin-1))/(multiplier));
	      // std::cout << "rowfrac : " << rowfrac << std::endl;
              val = (aMat(Oldrow_it, Oldcol_it) * (1-rowfrac)) + (aMat(Oldrow_it+1, Oldcol_it) * (rowfrac));
	    } else {
              val = aMat(Oldrow_it, Oldcol_it);
	    }
          }
	  // std::cout << val << std::endl;
          newMat1(row_it, col_it) = val/multiplier;
          // newMat1(row_it, col_it) = val;
        }
      }

	  // check position wrt mid-bin
          // check if you can get +1 OR -1 bin contents (depends on position with respect to mid-bin)
          // if possible, get and take weighted average
          // if not possible, take current bin value 

    } else { // midbin center IS between bins
      std::cout << "midbin center IS between bins " << std::endl;

      for (int row_it = 0; row_it < newMat1.rows(); row_it++) {
        for (int col_it = 0; col_it < newMat1.cols(); col_it++) {
          midbin = (multiplier/2.0) + 0.5;// 0.5 is "bin" center off-set
	  double val;
          double Oldrow_it = row_it/multiplier;
          double Oldcol_it = col_it;
	  // check if close to mid-bin
	  if ( ( ((row_it+1+0.6) > midbin) && ((row_it+1) < midbin) ) || ( ((row_it+1-0.6) < midbin) && ((row_it+1) > midbin) )  ) {
	    // std::cout << "equals - row_it+1 : " << row_it+1 << std::endl;
            val = aMat(Oldrow_it, Oldcol_it);
	  // check position wrt mid-bin
	  } else if ( ((row_it)%multiplier)+1 < midbin ) {
	    // std::cout << "less than - row_it+1 : " << row_it+1 << std::endl;
	    midbin -= 0.5;
	    // check if you can get N-1 bincontents
	    if ( (row_it)/multiplier != 0) {
            // if possible, get and take weighted average
              double rowfrac = std::abs(((row_it%multiplier)-(midbin-1))/(multiplier-1));
	      // std::cout << "rowfrac : " << rowfrac << std::endl;
              val = (aMat(Oldrow_it-1, Oldcol_it) * (rowfrac)) + (aMat(Oldrow_it, Oldcol_it) * (1-rowfrac));
	    } else {
              val = aMat(Oldrow_it, Oldcol_it);
	    }
	  // check position wrt mid-bin
	  } else if ( ((row_it)%multiplier)+1 > midbin ) {
	    // std::cout << "more than - row_it+1 : " << row_it+1 << std::endl;
	    midbin += 0.5;
	    // check if you can get N+1 bincontents
	    if ( (row_it)/multiplier != (aMat.rows()-1) ) {
            // if possible, get and take weighted average
              double rowfrac = std::abs(((row_it%multiplier)-(midbin-1))/(multiplier-1));
	      // std::cout << "rowfrac : " << rowfrac << std::endl;
              val = (aMat(Oldrow_it, Oldcol_it) * (1-rowfrac)) + (aMat(Oldrow_it+1, Oldcol_it) * (rowfrac));
	    } else {
              val = aMat(Oldrow_it, Oldcol_it);
	    }
          }
	  // std::cout << val << std::endl;
          newMat1(row_it, col_it) = val/multiplier;
          // newMat1(row_it, col_it) = val;
        }
      }
    }

    // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << " --- Matrix 1 --- " << std::endl;
    // std::cout << newMat1.format(CleanFmt) << std::endl;

    // manually average matrix over adjacent columns

    midbin = (multiplier/2.0) + 0.5;// 0.5 is "bin" center off-set
    std::cout << "midbin : " << midbin << std::endl;
    if ( std::ceil(midbin) == std::floor(midbin) ) { // midbin center is NOT between bins
      std::cout << "midbin center is NOT between bins " << std::endl;
      for (int row_it = 0; row_it < newMat2.rows(); row_it++) {
        for (int col_it = 0; col_it < newMat2.cols(); col_it++) {
	  double val;
          double Oldrow_it = row_it;
          double Oldcol_it = col_it/multiplier;
	  // check position wrt mid-bin
	  if ( ((col_it)%multiplier)+1 < midbin ) {
	    // std::cout << "less than - col_it+1 : " << col_it+1 << std::endl;
	    // check if you can get N-1 bincontents
	    if ( (col_it)/multiplier != 0) {
            // if possible, get and take weighted average
              double colfrac = std::abs(((col_it%multiplier)-(midbin-1))/(multiplier));
	      // std::cout << "colfrac : " << colfrac << std::endl;
              val = (newMat1(Oldrow_it, Oldcol_it-1) * (colfrac)) + (newMat1(Oldrow_it, Oldcol_it) * (1-colfrac));
	    } else {
              val = newMat1(Oldrow_it, Oldcol_it);
	    }
	  // check position wrt mid-bin
	  } else if ( ((col_it+1)%multiplier) == midbin ) {
	    // std::cout << "equals - col_it+1 : " << col_it+1 << std::endl;
	    val = newMat1(Oldrow_it, Oldcol_it);
	  // check position wrt mid-bin
	  } else if ( ((col_it)%multiplier)+1 > midbin ) {
	    // std::cout << "more than - col_it+1 : " << col_it+1 << std::endl;
	    // check if you can get N+1 bincontents
	    if ( (col_it)/multiplier != (newMat1.cols()-1) ) {
            // if possible, get and take weighted average
              double colfrac = std::abs(((col_it%multiplier)-(midbin-1))/(multiplier));
	      // std::cout << "colfrac : " << colfrac << std::endl;
              val = (newMat1(Oldrow_it, Oldcol_it) * (1-colfrac)) + (newMat1(Oldrow_it, Oldcol_it+1) * (colfrac));
	    } else {
              val = newMat1(Oldrow_it, Oldcol_it);
	    }
          }
	  // std::cout << val << std::endl;
          // newMat1(row_it, col_it) = val/multiplier;
          newMat2(row_it, col_it) = val;
        }
      }

	  // check position wrt mid-bin
          // check if you can get +1 OR -1 bin contents (depends on position with respect to mid-bin)
          // if possible, get and take weighted average
          // if not possible, take current bin value 

    } else { // midbin center IS between bins
      std::cout << "midbin center IS between bins " << std::endl;

      for (int row_it = 0; row_it < newMat2.rows(); row_it++) {
        for (int col_it = 0; col_it < newMat2.cols(); col_it++) {
          midbin = (multiplier/2.0) + 0.5;// 0.5 is "bin" center off-set
	  double val;
          double Oldrow_it = row_it;
          double Oldcol_it = col_it/multiplier;
	  // check if close to mid-bin
	  if ( ( ((col_it+1+0.6) > midbin) && ((col_it+1) < midbin) ) || ( ((col_it+1-0.6) < midbin) && ((col_it+1) > midbin) )  ) {
	    // std::cout << "equals - col_it+1 : " << col_it+1 << std::endl;
            val = newMat1(Oldrow_it, Oldcol_it);
	  // check position wrt mid-bin
	  } else if ( ((col_it)%multiplier)+1 < midbin ) {
	    // std::cout << "less than - col_it+1 : " << col_it+1 << std::endl;
	    midbin -= 0.5;
	    // check if you can get N-1 bincontents
	    if ( (col_it)/multiplier != 0) {
            // if possible, get and take weighted average
              // double colfrac = (col_it)/(midbin-1);
              double colfrac = std::abs(((col_it%multiplier)-(midbin-1))/(multiplier-1));
	      // std::cout << "colfrac : " << colfrac << std::endl;
              val = (newMat1(Oldrow_it, Oldcol_it-1) * (colfrac)) + (newMat1(Oldrow_it, Oldcol_it) * (1-colfrac));
	    } else {
              val = newMat1(Oldrow_it, Oldcol_it);
	    }
	  // check position wrt mid-bin
	  } else if ( ((col_it)%multiplier)+1 > midbin ) {
	    // std::cout << "more than - col_it+1 : " << col_it+1 << std::endl;
	    midbin += 0.5;
	    // check if you can get N+1 bincontents
	    if ( (col_it)/multiplier != (newMat1.cols()-1) ) {
            // if possible, get and take weighted average
              // double colfrac = (col_it)/(midbin-1);
              double colfrac = std::abs(((col_it%multiplier)-(midbin-1))/(multiplier-1));
	      // std::cout << "colfrac : " << colfrac << std::endl;
              val = (newMat1(Oldrow_it, Oldcol_it) * (1-colfrac)) + (newMat1(Oldrow_it, Oldcol_it+1) * (colfrac));
	    } else {
              val = newMat1(Oldrow_it, Oldcol_it);
	    }
          }
	  // std::cout << val << std::endl;
          // newMat2(row_it, col_it) = val/multiplier;
          newMat2(row_it, col_it) = val;
        }
      }
    }

    // fit matrix 
    // if ( fitScaleUp ) { 
    //   do something
    // }

    /*for (int col_it = 0; col_it < aMat.cols(); col_it++) {
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
    }*/

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

  std::pair<Eigen::MatrixXd, Eigen::VectorXd> constructLLS( const Eigen::MatrixXd &aTrueFluxMat, const Eigen::MatrixXd &aRecoFluxMat) {
    int Ebins = aTrueFluxMat.rows();

    Eigen::MatrixXd iFluxes = aRecoFluxMat.inverse();
    // Eigen::MatrixXd iFluxes = aRecoFluxMat;
    int matcols = (aTrueFluxMat.rows())*(aTrueFluxMat.rows());
    // int matrows = aTrueFluxMat.cols();
    int matrows = matcols;
    std::cout << "matcols : " << matcols << std::endl;
    std::cout << "matrows : " << matrows << std::endl;
    Eigen::MatrixXd LLSmat = Eigen::MatrixXd(matrows, matcols ); // rows/cols in reverse (weird) order 

    int newrowcounter = 0;
    for (int row_it = 0; row_it < aTrueFluxMat.cols(); row_it++) {
      for (int col_it = 0; col_it < aTrueFluxMat.rows(); col_it++) {
	int acounter = 0;
        for (int row_it2 = 0; row_it2 < aTrueFluxMat.cols(); row_it2++) {
          for (int col_it2 = 0; col_it2 < aTrueFluxMat.rows(); col_it2++) {
	    // std::cout << acounter << std::endl;
	    // std::cout << "i(row_it, col_it2)*j(row_it2, col_it) : i(" <<row_it<< ","<<col_it2<<")*j("<<row_it2<<","<<col_it<<")"<< std::endl;
	    LLSmat(newrowcounter, acounter ) = (iFluxes(row_it, col_it2)*aTrueFluxMat(row_it2, col_it)); // col_it2 leads assignment
	    acounter++;
          }
        }
	newrowcounter++;
      }
    }

    Eigen::VectorXd LLSvec = Eigen::VectorXd( matcols );
    // Eigen::VectorXd LLSvec = Eigen::VectorXd( aTrueFluxMat.rows() );

    /*for (int col_it = 0; col_it < aTrueFluxMat.rows(); col_it++) {
      for (int col_it2 = 0; col_it2 < aTrueFluxMat.rows(); col_it2++) {
	LLSvec((col_it*aTrueFluxMat.rows())+col_it2) = 1;
      }
    }*/

    int rcounter = 0;
    int ccounter = 0;
    // for (int row_it = 0; row_it < aTrueFluxMat.rows(); row_it++) {
    // }
    for (int row_it = 0; row_it < matcols; row_it++) {
	if (rcounter == ccounter) { 
	  LLSvec(row_it) = 1;
	} else {
	  LLSvec(row_it) = 0;
	}
	if (ccounter < (Ebins-1)) {
	  ccounter++;
	} else {
	  ccounter = 0;
	  rcounter++;
	}
    }

    return std::make_pair(LLSmat, LLSvec);
  }

  Eigen::MatrixXd deconstructLLS( const Eigen::VectorXd &SensingVector, int Ebins ) {

    Eigen::MatrixXd theMat = Eigen::MatrixXd( Ebins, Ebins);

    for (int col_it = 0; col_it < Ebins; col_it++) {
      for (int row_it = 0; row_it < Ebins; row_it++) {
	int v_it = (col_it*Ebins)+row_it;
	// std::cout << "v_it : " << v_it << std::endl;
	// std::cout << "(row_it, col_it) : (" << row_it <<","<<col_it<<")"<< std::endl;
	theMat(row_it, col_it) = SensingVector(v_it);
	// LLSmat(row_it, (col_it*aTrueFluxMat.rows())+col_it2) = iFluxes(col_it, row_it)*aTrueFluxMat(col_it2, row_it); // col_it2 leads assignment
      }
    }

    /*for (int row_it = 0; row_it < Ebins; row_it++) {
      for (int col_it = 0; col_it < Ebins; col_it++) {
	int v_it = (row_it*Ebins)+col_it;
	theMat(row_it, col_it) = SensingVector(v_it);
	// LLSmat(row_it, (col_it*aTrueFluxMat.rows())+col_it2) = iFluxes(col_it, row_it)*aTrueFluxMat(col_it2, row_it); // col_it2 leads assignment
      }
    }*/

    return theMat; 
  }

  Eigen::MatrixXd SolveSensingMatrix( const Eigen::MatrixXd &aTrueFluxMat, const Eigen::MatrixXd &aRecoFluxMat, double reg_param, double &res_norm, double &soln_norm, bool guidematrix = false ) {
  // Eigen::VectorXd runSolver(double reg_param, double &res_norm, double &soln_norm) {
  // }
    int Ebins = aTrueFluxMat.rows();

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> LLSout = constructLLS( aTrueFluxMat, aRecoFluxMat);
    Eigen::MatrixXd LLSoutmat = LLSout.first;
    Eigen::VectorXd LLSvec = LLSout.second;
    std::cout << "LLSmat.rows() : " << LLSoutmat.rows() << std::endl;
    std::cout << "LLSmat.cols() : " << LLSoutmat.cols() << std::endl;
    std::cout << "LLSvec.size() : " << LLSvec.size() << std::endl;
    Eigen::VectorXd SensingVector = Eigen::VectorXd::Zero(Ebins);

    bool use_reg = reg_param > 0;
    // size_t NFluxes = LLSoutmat.cols();
    // size_t NEqs = LLSmat.rows();
    // size_t NBins = NEqs - NFluxes;

    Eigen::MatrixXd LLSmat;

    bool EnforceSize = true; // should replace with fParams...
    if (EnforceSize) {
      int rowN = LLSoutmat.rows();
      LLSoutmat.conservativeResize(rowN+1, LLSoutmat.cols());
      int newrowN = rowN+1; 
      for (int col_it = 0; col_it < LLSoutmat.cols(); col_it++) {
        LLSoutmat(rowN, col_it) = 1; 
      }

      double forcedSize = std::sqrt(rowN);
      LLSvec.conservativeResize(rowN+1);
      // LLSvec(rowN) = rowN;
      LLSvec(rowN) = forcedSize;
    }


    if (use_reg) {
      std::cout << "Using regularised solver" << std::endl;
      int fluxbins = LLSoutmat.rows();
      // int newmatrows = LLSoutmat.rows()+LLSvec.size();
      // int newmatrows = LLSoutmat.rows()+LLSoutmat.cols();
      LLSmat = Eigen::MatrixXd::Zero(LLSoutmat.rows()+LLSvec.size(), LLSoutmat.cols());
      LLSmat.block(0, 0, LLSoutmat.rows(), LLSoutmat.cols()) = LLSoutmat;
      for (int r_iter = 0; r_iter < LLSvec.size(); r_iter++) {
        LLSmat(r_iter + fluxbins, r_iter) = reg_param;
      }
    } else {
      LLSmat = LLSoutmat;
    }

    SensingVector = runSolver(LLSmat, LLSvec, reg_param, res_norm, soln_norm);

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << " --- Sensing Vector --- " << std::endl;
    // std::cout << SensingVector.format(CleanFmt) << std::endl;
    // std::cout << " --- LLSmat --- " << std::endl;
    // std::cout << LLSmat.format(CleanFmt) << std::endl;
    // std::cout << " --- LLSvec --- " << std::endl;
    // std::cout << LLSvec.format(CleanFmt) << std::endl;
    std::cout << " --- aTrueFluxMat --- " << std::endl;
    std::cout << aTrueFluxMat.format(CleanFmt) << std::endl;
    std::cout << " --- aRecoFluxMat --- " << std::endl;
    std::cout << aRecoFluxMat.format(CleanFmt) << std::endl;

    Eigen::MatrixXd aSensingMat = deconstructLLS(SensingVector, Ebins);
    return aSensingMat;
  }

  Eigen::MatrixXd IterativeSolveSensingMatrix( const Eigen::MatrixXd &aTrueFluxMat, const Eigen::MatrixXd &aRecoFluxMat, double reg_param, double &res_norm, double &soln_norm, std::vector<double> &omega, bool guidematrix = false ) {
  // Eigen::VectorXd runSolver(double reg_param, double &res_norm, double &soln_norm) {
  // }
    int Ebins = aTrueFluxMat.rows();

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> LLSout = constructLLS( aTrueFluxMat, aRecoFluxMat);
    Eigen::MatrixXd LLSoutmat = LLSout.first;
    Eigen::VectorXd LLSvec = LLSout.second;
    std::cout << "LLSmat.rows() : " << LLSoutmat.rows() << std::endl;
    std::cout << "LLSmat.cols() : " << LLSoutmat.cols() << std::endl;
    std::cout << "LLSvec.size() : " << LLSvec.size() << std::endl;
    Eigen::VectorXd SensingVector = Eigen::VectorXd::Zero(Ebins);

    bool use_reg = reg_param > 0;
    size_t NFluxes = LLSoutmat.cols();
    // size_t NFluxes = FluxMatrix_Solve.cols();
    // size_t NEqs = FluxMatrix_Solve.rows();
    // size_t NBins = NEqs - NFluxes;
    
    Eigen::MatrixXd LLSmat;

    if (omega.size() == 0) {
      omega.assign(NFluxes, reg_param);
      /*if ( fParams.startCSequalreg ) {
        omega.assign(NFluxes, reg_param);
      } else {
        omega.assign(NFluxes, 1);
      }*/
    }

    if (use_reg) {
      std::cout << "Using iterative reweighting regularised solver" << std::endl;
      int fluxbins = LLSoutmat.rows();
      // int newmatrows = LLSoutmat.rows()+LLSvec.size();
      // int newmatrows = LLSoutmat.rows()+LLSoutmat.cols();
      LLSmat = Eigen::MatrixXd::Zero(LLSoutmat.rows()+LLSvec.size(), LLSoutmat.cols());
      LLSmat.block(0, 0, LLSoutmat.rows(), LLSoutmat.cols()) = LLSoutmat;
      for (int r_iter = 0; r_iter < LLSvec.size(); r_iter++) {
        LLSmat(r_iter + fluxbins, r_iter) = omega[r_iter];
      }
    } else {
      LLSmat = LLSoutmat;
    }

    SensingVector = runSolver(LLSmat, LLSvec, reg_param, res_norm, soln_norm);

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    // std::cout << " --- Sensing Vector --- " << std::endl;
    // std::cout << SensingVector.format(CleanFmt) << std::endl;
    // std::cout << " --- LLSmat --- " << std::endl;
    // std::cout << LLSmat.format(CleanFmt) << std::endl;
    // std::cout << " --- LLSvec --- " << std::endl;
    // std::cout << LLSvec.format(CleanFmt) << std::endl;
    // std::cout << " --- aTrueFluxMat --- " << std::endl;
    // std::cout << aTrueFluxMat.format(CleanFmt) << std::endl;
    // std::cout << " --- aRecoFluxMat --- " << std::endl;
    // std::cout << aRecoFluxMat.format(CleanFmt) << std::endl;

    std::vector<double> coeffvec(SensingVector.data(), SensingVector.data() +
                                        SensingVector.rows() * SensingVector.cols());
    omega = coeffvec;

    Eigen::MatrixXd aSensingMat = deconstructLLS(SensingVector, Ebins);
    return aSensingMat;
  }



  Eigen::VectorXd runSolver(Eigen::MatrixXd LLSmat, Eigen::VectorXd LLSvec, double reg_param, double &res_norm, double &soln_norm) {
    bool use_reg = reg_param > 0;
    size_t NFluxes = LLSmat.cols();
    size_t NEqs = LLSmat.rows();
    size_t NBins = NEqs - NFluxes;

    Eigen::VectorXd SensingVector;

    std::cout << "NFluxes : " << NFluxes << std::endl;
    std::cout << "NEqs : " << NEqs << std::endl;
    std::cout << "NBins : " << NBins << std::endl;
    std::cout << "LLSvec.size() : " << LLSvec.size() << std::endl;

    switch (fParams.algo_id) {
    case Params::kSVD: {
      // if (use_reg) {
        SensingVector =
            LLSmat.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(LLSvec);
      /*} else {
        SensingVector = LLSmat.topRows(NBins)
                            .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                            .solve(LLSvec.topRows(NBins));
      }*/
      break;
    }
    case Params::kQR: {
      // if (use_reg) {
        SensingVector = LLSmat.colPivHouseholderQr().solve(LLSvec);
      /*} else {
        SensingVector =
            LLSmat.topRows(NBins).colPivHouseholderQr().solve(
                LLSvec.topRows(NBins));
      }*/
      break;
    }
    case Params::kNormal: {
      // if (use_reg) {
        SensingVector = (LLSmat.transpose() * LLSmat)
                            .ldlt()
                            .solve(LLSmat.transpose() * LLSvec);
      /*} else {
        SensingVector = (LLSmat.transpose() * LLSmat)
                            .topRows(NBins)
                            .ldlt()
                            .solve(LLSmat.topRows(NBins).transpose() *
                                   LLSvec.topRows(NBins));
      }*/
      break;
    }
    case Params::kInverse: {
      if (use_reg) {
        std::cout << "Inverse solve" << std::endl;
        SensingVector = ((LLSmat.topRows(NBins).transpose() *
                          LLSmat.topRows(NBins)) +
                         LLSmat.bottomRows(NFluxes).transpose() *
                             LLSmat.bottomRows(NFluxes))
                            .inverse() *
                        LLSmat.topRows(NBins).transpose() *
                        LLSvec.topRows(NBins);
      } else {
        SensingVector = (LLSmat.transpose() *
                         LLSmat)
                            .inverse() *
                        LLSmat.transpose() *
                        LLSvec;
      }
      break;
    }
    case Params::kCOD: {
      // if (use_reg) {
        SensingVector = LLSmat.completeOrthogonalDecomposition().solve(LLSvec);
      /*} else {
        SensingVector =
            LLSmat.topRows(NBins).completeOrthogonalDecomposition().solve(
                LLSvec.topRows(NBins));
      }*/
      break;
    }
    case Params::kConjugateGradient: {
      // SparseMatrix<double> = LLSmat; // setting to sparsematrix - may not work - probably not optimal for runtimes
      // BiCGSTAB<SparseMatrix<double> > solver; // template
      //BiCGSTAB<Eigen::MatrixXd> solver; // 
      // ConjugateGradient<Eigen::MatrixXd>, Lower|Upper solver;
      // Eigen::ConjugateGradient<Eigen::MatrixXd, Lower|Upper> solver;

      Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> solver;
      std::cout << "ConjugateGradient solve" << std::endl;
      if (use_reg) {
        solver.compute((LLSmat.topRows(NBins).transpose() *
                        LLSmat.topRows(NBins)) +
                        LLSmat.bottomRows(NFluxes).transpose() *
                        LLSmat.bottomRows(NFluxes));

        SensingVector = solver.solve(LLSmat.topRows(NBins).transpose() *
                                     LLSvec.topRows(NBins));
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;
      } else {
        solver.compute(LLSmat.transpose() *
                        LLSmat);

        SensingVector = solver.solve(LLSmat.transpose() *
                        LLSvec);
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;
      }
      break;
    }
    }
    if (!SensingVector.rows()) {
      res_norm = 0;
      soln_norm = 0;
      return SensingVector;
    }

    if (use_reg) {
      res_norm = ((LLSmat.topRows(NBins) * SensingVector) -
                   LLSvec.topRows(NBins))
                   .squaredNorm();
    } else {
      res_norm = ((LLSmat * SensingVector) -
                   LLSvec)
                   .squaredNorm();
    }
    soln_norm = 0;
    if (reg_param > 0) {
      soln_norm =
          (LLSmat.bottomRows(NFluxes) * SensingVector / reg_param)
              .squaredNorm();
    }

    if ( isnan(res_norm) ) {
        std::cerr << "[ERROR] : NaN res norm found. " << std::endl;
        std::cerr << SensingVector << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if ( isnan(soln_norm) ) {
        std::cerr << "[ERROR] : NaN soln norm found. " << std::endl;
        std::cerr << SensingVector << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return SensingVector;
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

  void WriteMatrix2D(TDirectory *td, int nEbinsX, int nEbinsY, Eigen::MatrixXd aMat, TString prefix) {
    TString hname = TString::Format("%s", prefix.Data());
    TH2D *aHist = new TH2D (hname, hname, nEbinsX, 0.0, 10.0, nEbinsY, 0.0, 10.0);

    std::cout << "Filling histogram with " << aHist->GetXaxis()->GetNbins() 
              << " bins from matrix with " << aMat.rows() << " rows and " 
              << aMat.cols() << " cols."
              << std::endl;
    // size_t idx = 0;
    // for (Int_t x_it = bin_offset; x_it < aHist->GetXaxis()->GetNbins(); ++x_it) {
    /*double v = (idx >= vals.rows()) ? 0 : vals(idx);
        aHist->SetBinContent(x_it + 1, v);
        aHist->SetBinError(x_it + 1, 0);
        idx++;*/
    // }
    for (int row_it = 0; row_it < aMat.rows(); row_it++) {
      for (int col_it = 0; col_it < aMat.cols(); col_it++) {
        aHist->SetBinContent(col_it+1, row_it+1, aMat(row_it, col_it) );
        aHist->SetBinError(col_it+1, row_it+1, 0);
      }
    }
    aHist->SetDirectory(td);
    aHist->Write();

    // Reset flow bins
    aHist->ClearUnderflowAndOverflow();
    // Below - 1D case :
    /*aHist->SetBinContent(0, 0);
    aHist->SetBinError(0, 0);
    aHist->SetBinContent(aHist->GetXaxis()->GetNbins() + 1, 0);
    aHist->SetBinError(aHist->GetXaxis()->GetNbins() + 1, 0);
    aHist->SetBinContent(aHist->GetXaxis()->GetNbins() + 1, 0);
    aHist->SetBinError(aHist->GetXaxis()->GetNbins() + 1, 0);*/

    return;
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
