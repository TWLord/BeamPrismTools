#include "FluxLinearSolver_Standalone.hxx"

#include "TGraph.h"
// extra stuff for plotting
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TLine.h"

// Global Var Defs
std::string OutputFile = "ERecoSolver.root";

std::string NDFile, FDFile, FDHist, WFile;

std::vector<std::string> NDHists;

std::string OARange;
std::string NomOARange = "";

int NEnuBinMerge = 0;

int method = 1;
int weightmethod = 1;
size_t nsteps = 20;

double OutOfRangeChi2Factor = 0.1;
double NominalRegFactor = 1E-9; 
double BCRegFactor = 1;
double CSNorm = -8;
double StabilityFactor = -3;
double FitRangeLow = 0.5, FitRangeHigh = 10.0;
double CurrentRangeLow = 75, CurrentRangeHigh = 350;
std::vector<std::vector<std::pair<double, double>>> CurrentRangesplit;
// std::vector<std::pair<double, double>> CurrentRangesplit = {};
double coeffMagLimit = 0;
int leastNCoeffs= 0;
double NominalCurrent = 293; 
// int NominalHist = 4;
size_t NominalHist = 4;


//
//
//

// Helper Fctns
void handleOpts(int argc, char const *argv[]) {
  int opt = 1;
  while (opt < argc) {
    if (std::string(argv[opt]) == "-o") {
      OutputFile = argv[++opt];
    } else if (std::string(argv[opt]) == "-N") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      // if (params.size() != 2) {
      //   std::cout << "[ERROR]: Recieved " << params.size()
      //             << " entrys for -i, expected 2." << std::endl;
      //   exit(1);
      // }
      NDFile = params[0];
      NDHists = ParseToVect<std::string>(params[1], ":");
    } else if (std::string(argv[opt]) == "-F") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -i, expected 2." << std::endl;
        exit(1);
      }
      FDFile = params[0];
      FDHist = params[1];
    } else if (std::string(argv[opt]) == "-W") {
      WFile = std::string(argv[++opt]);
    } else if (std::string(argv[opt]) == "-WM") {
      weightmethod = str2T<int>(argv[++opt]);
      if ((weightmethod < 1) || (weightmethod > 2)) {
        std::cout
            << "[WARN]: Invalid option for weighting method, defaulting to total flux weighting."
            << std::endl;
        weightmethod = 1;
      }
    } else if (std::string(argv[opt]) == "-M") {
      method = str2T<int>(argv[++opt]);
      if ((method < 1) || (method > 9)) {
        std::cout
            << "[WARN]: Invalid option for solving method, defaulting to QR."
            << std::endl;
        method = 2;
      }
    } else if (std::string(argv[opt]) == "-FR") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -FR, expected 2." << std::endl;
        exit(1);
      }
      FitRangeLow = str2T<double>(params[0]);
      FitRangeHigh = str2T<double>(params[1]);
    } else if (std::string(argv[opt]) == "-CR") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -CR, expected 2." << std::endl;
        exit(1);
      }
      CurrentRangeLow = str2T<double>(params[0]);
      CurrentRangeHigh = str2T<double>(params[1]);
    } else if (std::string(argv[opt]) == "-CRS") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ":");
      for (std::string v1 : params) {
	std::vector<std::string> sub_vect = ParseToVect<std::string>(v1, ",");
	std::vector<std::pair<double, double>> sub_pair;
        for (std::string v2 : sub_vect) {
	  std::vector<std::string> subsub_vect = ParseToVect<std::string>(v2, "_");
	  if (subsub_vect.size() != 2) {
            std::cout << "[ERROR]: Recieved " << subsub_vect.size()
                      << " entrys for an element in -CRS, expected 2." << std::endl;
            exit(1);
	  }
	  std::pair<double, double> subsub_pair = { str2T<double>(subsub_vect[0]), str2T<double>(subsub_vect[1]) };
	  sub_pair.emplace_back(subsub_pair);
	}
        CurrentRangesplit.emplace_back(sub_pair);
      }
    } else if (std::string(argv[opt]) == "-Nom") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -i, expected 2." << std::endl;
        exit(1);
      }
      NominalHist = str2T<int>(params[0]);
      NominalCurrent = str2T<double>(params[1]);
      if ( NominalCurrent < CurrentRangeLow ||
		 NominalCurrent > CurrentRangeHigh ) {
        std::cout << "[ERROR]: Nominal Current " << NominalCurrent
		  << " not in interval (" << CurrentRangeLow 
		  << "," << CurrentRangeHigh << ")" << std::endl;
	exit(1);
      }
      if ( NominalHist > NDHists.size() ) {
        std::cout << "[ERROR]: Nominal Hist = " << NominalHist
		  << " larger than hists passed to fit" << std::endl; 
	exit(1);
      }
    } else if (std::string(argv[opt]) == "-MX") {
      NEnuBinMerge = str2T<int>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-OA") {
      OARange = std::string(argv[++opt]);
    } else if (std::string(argv[opt]) == "-NOA") {
      NomOARange = std::string(argv[++opt]);
    } else if (std::string(argv[opt]) == "-OR") {
      OutOfRangeChi2Factor = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-CML") {
      coeffMagLimit = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-CNL") {
      leastNCoeffs= str2T<int>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-NSTEPS") {
      nsteps = str2T<int>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-NRF") {
      NominalRegFactor = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-RF") {
      BCRegFactor = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-SF") {
      StabilityFactor = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-CSnorm") {
      CSNorm = str2T<double>(argv[++opt]);
    } else if ((std::string(argv[opt]) == "-?") ||
               (std::string(argv[opt]) == "--help")) {
      SayUsage(argv);
      exit(0);
    } else {
      std::cout << "[ERROR]: Unknown option: \"" << argv[opt] << "\""
                << std::endl;
      SayUsage(argv);
      exit(1);
    }
    opt++;
  }
  if (coeffMagLimit && leastNCoeffs) { 
      std::cout << "[ERROR]: Can only use either magnitude coeff limit OR number coefficient limit: \""
                << std::endl;
      SayUsage(argv);
      exit(1);
  }
}

// main
int main(int argc, char const *argv[]) {
  TH1::SetDefaultSumw2();
  handleOpts(argc, argv);
  
  ERecoSolver ers;
  ERecoSolver::Params p = ERecoSolver::GetDefaultParams();

  // Set Params here 
  // p. .....

  // ers.Initialize();


  // ers.ComputeMatrix();
  
  // ers.PlotMatrix();
  // ers.Write(f); // add any other to-write external vals
  f->Write();
}
