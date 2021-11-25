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

std::string OutputFile = "FluxLinearSolver.root";

std::string NDFile, FDFile, FDHist, WFile, BCTree, BCBranches;

std::vector<std::string> NDHists;

std::string OARange;
std::string NomOARange = "";

int NEnuBinMerge = 0;

bool scalebyE = false;
bool startcsequalreg = true;

int method = 1;
int weightmethod = 1;
int coeffmethod = 1;
size_t nsteps = 20;

double OutOfRangeChi2Factor = 1.0;
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

void SayUsage(char const *argv[]) {
  std::cout << "Runlike: " << argv[0]
            << " -N <NDFluxFile,NDFluxHistName> -F <FDFluxFile,FDFluxHistName> "
	       "[ -W <FDWeightFile> -WM <1:TotalFlux, 2:MaxFlux> -o Output.root "
               "-M <1:SVD, 2:QR, 3:Normal, 4:Inverse, 5:COD, 6:ConjugateGradient, 7:LeastSquaresConjugateGradient, 8: BiCGSTAB, 9: BiCGSTAB w/ last sol'n guess> "
               "-CoeffM <1:CNLSolveOnce, 2:CNLSolve, 3:OrthogSolve, 4:OrthogGS, 5:CSSolve > "
               "-ScaleE <1:true/0:false> -CSeqreg <1:true/0:false> "
               "-MX <NEnuBinMerge> -OR OutOfRangeChi2Factor -RF BeamConfigsRegFactor "
               "-CML <CoeffMagLowerBound> -CNL <CoefficientNumberLimit> "
	       "-B <ConfTree,ConfBranches> -Nom <NominalHist(number),NominalCurrent> "
	       "-FR <FitRangeLow, FitRangeHigh> -CR <CurrentRangeLow,CurrentRangeHigh> "
	       "-CRsplit <CR1Low,CR1High:CR2Low,CR2High:> "
	       "-OA <OffAxisLow_OffAxisHigh:BinWidth,....,OffAxisLow_OffAxisHigh:BinWidth> ]"
	       "-NOA <NOALow_NOAHigh:BinWidth,....,NOALow_NOAHigh:BinWidth> ]"
            << std::endl;
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
    } else if (std::string(argv[opt]) == "-B") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -i, expected 2." << std::endl;
        exit(1);
      }
      BCTree = params[0];
      BCBranches = params[1];
//      BCTree = "ConfigTree";
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
    } else if (std::string(argv[opt]) == "-CoeffM") {
      coeffmethod = str2T<int>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-ScaleE") {
      scalebyE = str2T<bool>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-CSeqreg") {
      startcsequalreg = str2T<bool>(argv[++opt]);
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

int main(int argc, char const *argv[]) {
  TH1::SetDefaultSumw2();
  handleOpts(argc, argv);

  FluxLinearSolver fls;
  FluxLinearSolver::Params p = FluxLinearSolver::GetDefaultParams();

  p.algo_id = FluxLinearSolver::Params::Solver(method);

  // Eigen LS solver
  // p.algo_id = FluxLinearSolver::Params::kSVD; // Slowest but most accurate
  // p.algo_id =
  //     FluxLinearSolver::Params::kQR; // Middle, includes pre-conditioning
  // p.algo_id = FluxLinearSolver::Params::kNormal; // Fastest but least
  // accurate
  // p.algo_id =
  //     FluxLinearSolver::Params::kInverse; // Middle, includes
  //     pre-conditioning

  // Out of fit range behavior
  // p.OORMode = FluxLinearSolver::Params::kGaussianDecay; // To left/right of last
  p.OORMode = FluxLinearSolver::Params::kWeightOnly; // Only uses OORF weightin for ranges, no appended gaussians 
  //     'fit'
  // bin decay gaussianly
  // p.OORMode =
  //     FluxLinearSolver::Params::kIgnore; // Throw away bins outside of the
  //     range
  // p.OORMode = FluxLinearSolver::Params::kZero; // Set the target outside of
  // the range to zero

  // Out of fit range sides to care about
  // p.OORSide = FluxLinearSolver::Params::kLeft; // Try and fit low energy side
  p.OORSide = FluxLinearSolver::Params::kBoth; // Try and fit both low and high energy side
  // p.OORSide = Params::kRight; // Try and fit high energy side

  // Rate of gaussian decay for p.OORMode == Params::kGaussianDecay
  p.ExpDecayRate = 3;

  p.CurrentRange = {CurrentRangeLow, CurrentRangeHigh};
  p.CurrentRangeSplit = CurrentRangesplit;
  p.NominalFlux = {NominalHist, NominalCurrent};

  // Chi2 factor out of fit range
  p.OORFactor = pow(OutOfRangeChi2Factor, 0.5);
  p.coeffMagLower = coeffMagLimit;
  p.LeastNCoeffs = leastNCoeffs;
  p.FitBetweenFoundPeaks = false;
  // p.FitBetween = {0.5,10.0};
   p.FitBetween = {FitRangeLow,FitRangeHigh};
  p.MergeENuBins = NEnuBinMerge;
  p.MergeOAPBins = 0;
  // Use 0.5 m flux windows between -0.25 m and 32.5 m (65)
  //p.OffAxisRangesDescriptor = "-1.45_37.55:0.1";
  p.OffAxisRangesDescriptor = OARange;
  p.NominalOffAxisRangesDescriptor = NomOARange;
  /*p.NominalOffAxisRangesDescriptor = OARange;
  if ( NomOARange != "" ) {
  	p.NominalOffAxisRangesDescriptor = NomOARange;
  }*/
  p.WFile = WFile;
  p.WeightMethod = FluxLinearSolver::Params::Weighting(weightmethod);
  //p.OffAxisRangesDescriptor = "-1.45_32.45:0.1,37.45_37.55:0.1";

  // p.startCSequalreg = true;
  // p.OrthogSolve = true;

  // Weights fluxes by E for solving
  p.ScaleByE = scalebyE;
  // For CSSolve, sets starting reg factor for iterative CS solver equal to nominal reg factor 
  p.startCSequalreg = startcsequalreg;
  // p.InvCovWeighting = 0.8;

  p.CoeffMethod = FluxLinearSolver::Params::CoeffReducer(coeffmethod);
 

  int ncoeffs = 0;
  if (nsteps == 1) {
    p.CoeffMethod = FluxLinearSolver::Params::kCNLSolveOnce;
  }

  fls.Initialize(p, ncoeffs, {NDFile, NDHists}, {FDFile, FDHist}, {BCTree, BCBranches}, true);

  // std::array<double, 6> OscParameters{0.297,   0.0214,   0.534,
  //                                     7.37E-5, 2.539E-3, 0};
  // // numu disp
  // std::pair<int, int> OscChannel{14, 14};
  //
  // // Defaults to DUNE baseline;
  // fls.OscillateFDFlux(OscParameters, OscChannel);

  // size_t nsteps = 20;
  // double start = -18;
  

  if (p.CoeffMethod == FluxLinearSolver::Params::kOrthogGS) {
    std::cout << " ---- DOING OrthogGS SOLVE ----" << std::endl;
    std::string GSFile = OutputFile.substr(0, OutputFile.size()-5)+"_ortho.root";
    TFile *orthof = CheckOpenFile(GSFile, "RECREATE");
    // fls.WriteOrthogs(f);
    fls.doGS(orthof);
    orthof->Write();
  }

  // if (p.OrthogSolve) {
  if (p.CoeffMethod == FluxLinearSolver::Params::kOrthog) {
    std::cout << " ---- DOING Orthog SOLVE ----" << std::endl;
    double soln_norm, res_norm;
    // double reg_exp = BCRegFactor;
    double reg_exp = NominalRegFactor;
    // double reg_exp = -9;
    // write function that runs solve for nominal input fluxes,
    // sorts based on QR orthog projection, then re-calls solve.. 
    // but can also solve for nominal + additional fluxes.. 
    // use flux_reduced architecture in header file, there is now a param,
    // OrthogSolve which, if set, doesn't change FluxMatrix_Solve.. 
    // However you will need to re-set regularisation manually. :/   
    fls.doResidualOrthog( pow(10, reg_exp), 1, res_norm, soln_norm, OutputFile );
    //
    // then can call function to project all fluxes in fluxmatrix_solve
    // onto residual vector, (bf - target). This is just dot product of
    // each flux (col) with residual / residual dot residual ( * residual? )
    // fls.WriteOrthogSolve(OutputFile, res_norm, soln_norm);
  }

  if (p.CoeffMethod == FluxLinearSolver::Params::kCSSolve) {
    std::cout << " ---- DOING CS SOLVE ----" << std::endl;

    // looking at coeff removal below coefflim size
    double coefflim = 1E-9;
    TGraph coeffs(nsteps);
    std::cout << "ncoeffs : " << ncoeffs << std::endl;
    TH2D *CoeffChange =
       new TH2D("CoeffChange2D", "Coeff Change with steps", ncoeffs, 0, ncoeffs, nsteps+1, 0, nsteps+1);
    TH2D *WeightChange =
       new TH2D("WeightChange2D", "Weighting Change with steps", ncoeffs, 0, ncoeffs, nsteps+1, 0, nsteps+1);

    TGraph lcurve(nsteps);
    // double reg_exp = -9;
    // double reg_exp = BCRegFactor;
    double reg_exp = NominalRegFactor;
    std::vector<double> omega;
    double soln_norm, res_norm;

    std::vector<double> eta_hat, rho_hat;
    for (size_t l_it = 0; l_it < nsteps; ++l_it) {

      /*for (int i = 0; i < omega.size(); i++) {
        // std::cout << omega[i] << std::endl;
	WeightChange->SetBinContent( i+1, l_it+1, omega[i]);
      }*/

      fls.CompressedSensingSolve(pow(10, reg_exp), omega, res_norm, soln_norm);
      std::cout << "soln_norm : " << soln_norm << std::endl;
      std::cout << "res_norm : " << res_norm << std::endl;
      eta_hat.push_back(log(soln_norm));
      rho_hat.push_back(log(res_norm));

      lcurve.SetPoint(l_it, rho_hat.back() / 2.0, eta_hat.back() / 2.0);

      // std::cout << "\n --------------- Solve Coeffs --------------- " << std::endl;
      double largecoeffs = 0;
      double coeffsum = 0;

      for (size_t i = 0; i < omega.size(); i++) {
        // std::cout << omega[i] << std::endl;
	CoeffChange->SetBinContent( i+1, l_it+1, omega[i]);
	if (omega[i] > coefflim) {
	  largecoeffs+=1;
	  coeffsum += omega[i];
	}
      }
      coeffs.SetPoint(l_it, coeffsum, largecoeffs ); 

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

    TFile *f = CheckOpenFile(OutputFile, "RECREATE");
    fls.WriteCS(f, res_norm, soln_norm);

    // This step is to filter coefficient results
    // possibly find largest coeff discontinuity and set this as upper lim?
    int w_it = 0; // need to do this in header file function to access nominal flux indexing
    // or to turn off coeffs by removing columns like in removencoeffs funcs? 
    for (double &weight : omega) {
      if ( std::abs (weight) >= coefflim ) {
        weight = std::abs( pow(10, CSNorm) /((pow(10,-CSNorm)*weight) + pow(10,StabilityFactor)) );
      }
      else {
	weight = 1/(pow(10,1E-100));
      }
      WeightChange->SetBinContent( w_it+1, nsteps+1, weight);
      w_it++;
    }

    fls.CompressedSensingSolveLast(pow(10, reg_exp), omega, res_norm, soln_norm, pow(10, CSNorm), pow(10, StabilityFactor));

    w_it = 0;
    for (double &weight : omega) {
      CoeffChange->SetBinContent( w_it+1, nsteps+1, weight);
      weight = std::abs( pow(10, CSNorm) /((pow(10,-CSNorm)*weight) + pow(10,StabilityFactor)) );
      WeightChange->SetBinContent( w_it+1, nsteps+1, weight);
      w_it++;
    }

    // TFile *f = CheckOpenFile(OutputFile, "RECREATE");
    fls.Write(f, res_norm, soln_norm);
    lcurve.Write("CS_lcurve");
    coeffs.Write("coeffchange");
    CoeffChange->Write();
    WeightChange->Write();
    // kcurve.Write("kcurve");
    f->Write();

  } else if (p.CoeffMethod == FluxLinearSolver::Params::kCNLSolveOnce) {
    std::cout << " ---- DOING CNL SOLVE ONCE ----" << std::endl;
    double soln_norm=0, res_norm=0;
    // double input_reg = pow(10,-9);
    double input_reg = NominalRegFactor; 
    fls.SolveLast(input_reg, BCRegFactor, res_norm, soln_norm);

    TFile *f = CheckOpenFile(OutputFile, "RECREATE");
    fls.Write(f, res_norm, soln_norm);
    f->Write();

  } else if (p.CoeffMethod == FluxLinearSolver::Params::kCNLSolve) {
    std::cout << " ---- DOING CNL SOLVE ----" << std::endl;

    double start = -10;
    double end = -7;
    // double end = -2;
    TGraph lcurve(nsteps);
    TGraph kcurve(nsteps - 8);
    double step = double(end - start) / double(nsteps);

    std::vector<double> eta_hat, rho_hat;
    for (size_t l_it = 0; l_it < nsteps; ++l_it) {
      std::cout << "\n\n ------ Step : " << l_it << " ------ " << std::endl;
      double reg_exp = start + double(l_it) * step;
      // Passed parameter is regularization factor, should scan for the best one,
      double soln_norm, res_norm;
      fls.Solve(pow(10, reg_exp), BCRegFactor, res_norm, soln_norm);
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

    double soln_norm=0, res_norm=0;
    fls.SolveLast(best_reg, BCRegFactor, res_norm, soln_norm);

    TFile *f = CheckOpenFile(OutputFile, "RECREATE");
    fls.Write(f, res_norm, soln_norm);
    lcurve.Write("lcurve");
    kcurve.Write("kcurve");
    f->Write();
  }
}
