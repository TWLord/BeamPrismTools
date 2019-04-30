#include "FluxFitter.hxx"

#include "SliceConfigTreeReader.hxx"

#include "ROOTUtility.hxx"
#include "StringParserUtility.hxx"

#include "Fit/Fitter.h"
#include "Math/Factory.h"
#include "Math/Functor.h"

#include <algorithm>
#include <chrono>
#include <functional>

// #define DEBUG_LHOOD

void FluxFitter::Initialize(FluxFitterOptions const &ffo) {
  fFitterOpts = ffo;
  std::cout << "[INFO]: Initializing fit with: " << DumpFluxFitterOptions(ffo)
            << std::endl;
}

void FluxFitter::InitializeFlux(
    FluxFitter::FluxFitterOptions const &ffo,
    FluxTargetLikelihood::FluxTargetOptions const &fto,
    bool ExpectTargetLater) {
  Initialize(ffo);
  std::cout << "[INFO]: Initializing Flux fit with: "
            << DumpFluxTargetOptions(fto) << std::endl;
  fFitterOpts.IsGauss = false;
  std::unique_ptr<FluxTargetLikelihood> fluxLHood =
      std::make_unique<FluxTargetLikelihood>();

  fluxLHood->Initialize(fto);
  LHoodEval = std::move(fluxLHood);

  if (!ExpectTargetLater) {
    if (!fFitterOpts.InputTargetFluxFile.size() ||
        !fFitterOpts.InputTargetFluxName.size()) {
      std::cout << "[ERROR]: No target flux file specified." << std::endl;
      throw;
    }

    tgtFlux = GetHistogram_uptr<TH1D>(fFitterOpts.InputTargetFluxFile,
                                      fFitterOpts.InputTargetFluxName);

    SetTargetFlux(tgtFlux.get(), true);
  }
}
void FluxFitter::InitializeGauss(
    FluxFitter::FluxFitterOptions const &ffo,
    GausTargetLikelihood::GausTargetOptions const &gto) {

  Initialize(ffo);

  std::cout << "[INFO]: Initializing Gaus fit with: "
            << DumpGausTargetOptions(gto) << std::endl;

  fFitterOpts.IsGauss = true;
  std::unique_ptr<GausTargetLikelihood> gausLHood =
      std::make_unique<GausTargetLikelihood>();

  gausLHood->Initialize(gto);

  LHoodEval = std::move(gausLHood);

  InitializeCoefficientGuess();
}

void FluxFitter::InitializeCoefficientGuess() {

  Coefficients.clear();
  size_t NFluxes = LHoodEval->GetNFluxes();
  std::fill_n(std::back_inserter(Coefficients), NFluxes, 0);

  if (fFitterOpts.InpCoeffFile.size()) {
    if (fFitterOpts.InpCoeffDir.size() &&
        (fFitterOpts.InpCoeffDir.back() != '/')) {
      fFitterOpts.InpCoeffDir += "/";
    }

    std::cout << "[INFO]: Reading input previous fit results." << std::endl;
    SliceConfig sc(fFitterOpts.InpCoeffFile, fFitterOpts.InpCoeffDir);

    std::vector<std::pair<double, double>> inpXRanges = sc.GetXRanges();
    std::vector<double> inpCoeffs = sc.GetCoeffs();

    if (inpXRanges.size() != LHoodEval->fFSO.XRanges.size()) {
      std::cout << "[ERROR]: Input fit result had " << inpXRanges.size()
                << " off-axis slices, but here we have found "
                << LHoodEval->fFSO.XRanges.size()
                << ". Input results are incompatible." << std::endl;
      throw;
    }

    for (size_t f_it = 0; f_it < inpXRanges.size(); ++f_it) {
      if ((fabs(inpXRanges[f_it].first -
                (100. * LHoodEval->fFSO.XRanges[f_it].first)) > 1E-5) ||
          (fabs(inpXRanges[f_it].second -
                (100. * LHoodEval->fFSO.XRanges[f_it].second)) > 1E-5)) {
        std::cout << "\tFlux window[" << f_it << "] = {"
                  << inpXRanges[f_it].first << " -- " << inpXRanges[f_it].second
                  << "}, Coeff = " << inpCoeffs[f_it] << std::endl;
        std::cout << "[ERROR]: Here we found Flux window[" << f_it << "] = {"
                  << LHoodEval->fFSO.XRanges[f_it].first << " -- "
                  << LHoodEval->fFSO.XRanges[f_it].second
                  << "}. Input results are incompatible." << std::endl;
        throw;
      }
      Coefficients[f_it] = inpCoeffs[f_it];
    }

  } else {
    size_t flux_with_closest_peak_it = LHoodEval->GetNearestPeakingFluxIndex();

    for (size_t flux_it = 0; flux_it < NFluxes; flux_it++) {
      if (std::abs(long(flux_it) - long(flux_with_closest_peak_it)) ==
          1) { // Neighbours are negative
        Coefficients[flux_it] = -0.35;
      } else if (std::abs(long(flux_it) - long(flux_with_closest_peak_it)) == 0) {
        Coefficients[flux_it] = 1;
      } else { // Others start free
        Coefficients[flux_it] = 0;
      }
    }
  }
}

bool FluxFitter::Fit() {

  size_t NFluxes = LHoodEval->GetNFluxes();

  if (!fMinimizer) {
    fMinimizer = std::unique_ptr<ROOT::Math::Minimizer>(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));
  }

  fMinimizer->SetMaxFunctionCalls(fFitterOpts.MaxLikelihoodCalls);
  fMinimizer->SetTolerance(fFitterOpts.MINUITTolerance);
  fMinimizer->SetPrintLevel(3);

  for (size_t flux_it = 0; flux_it < NFluxes; flux_it++) {
    std::stringstream ss("");
    ss << "fluxpar" << flux_it;

    fMinimizer->SetVariable(flux_it, ss.str().c_str(), Coefficients[flux_it],
                            0.1);
    fMinimizer->SetVariableLimits(flux_it, -fFitterOpts.CoeffLimit,
                                  fFitterOpts.CoeffLimit);
  }

  auto lap = std::chrono::high_resolution_clock::now();
  ROOT::Math::Functor fcn(
      [&](double const *coeffs) {
        if (LHoodEval->GetNLHCalls() && !(LHoodEval->GetNLHCalls() % 5000)) {
          auto diff = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::high_resolution_clock::now() - lap);
          lap = std::chrono::high_resolution_clock::now();
          std::cout << "[INFO]: Call: " << LHoodEval->GetNLHCalls()
                    << ", Last 5k calls took " << diff.count() << "s"
                    << std::endl;
          std::cout << "[INFO]: Likelihood state: " << LHoodEval->State()
                    << std::endl;
        }
#ifdef DEBUG_LHOOD
        std::cout << "[INFO]: Step " << fcn_call << std::endl;
        for (size_t c = 0; c < NFluxes; ++c) {
          std::cout << "\t c[" << c << "] = " << coeffs[c] << std::endl;
        }
#endif
        double lh = LHoodEval->GetLikelihood(coeffs);
#ifdef DEBUG_LHOOD
        std::cout << "LH = " << lh << std::endl;
#endif

        return lh;
      },
      NFluxes);

  fMinimizer->SetFunction(fcn);

  fMinimizer->Minimize();
  int fitstatus = fMinimizer->Status();
  if (fitstatus) {
    std::cout << "[WARN]: Failed to find minimum (STATUS: " << fitstatus << ")."
              << std::endl;
  }

  for (size_t flux_it = 0; flux_it < NFluxes; flux_it++) {
    Coefficients[flux_it] = fMinimizer->X()[flux_it];
  }

  return (!fitstatus);
}

void FluxFitter::SetCoefficients(std::vector<double> coeffs_new) {
  Coefficients = std::move(coeffs_new);
}
void FluxFitter::SetTargetFlux(TH1D const *tgt, bool GuessCoefficients) {
  if (fFitterOpts.IsGauss) {
    std::cout << "[ERROR]: Attempted to SetTargetFlux on a FluxFitter that is "
                 "configured to fit gaussian fluxes."
              << std::endl;
    throw;
  }

  dynamic_cast<FluxTargetLikelihood *>(LHoodEval.get())->SetTargetFlux(tgt);

  if (GuessCoefficients) {
    InitializeCoefficientGuess();
  }
}
void FluxFitter::SetGaussParameters(double GaussC, double GaussW,
                                    bool GuessCoefficients) {
  if (!fFitterOpts.IsGauss) {
    std::cout << "[ERROR]: Attempted to SetGaussParameters on a FluxFitter "
                 "that is configured to fit target fluxes."
              << std::endl;
    throw;
  }
  dynamic_cast<GausTargetLikelihood *>(LHoodEval.get())
      ->SetGaussParameters(GaussC, GaussW);

  if (GuessCoefficients) {
    InitializeCoefficientGuess();
  }
}

void FluxFitter::Write(TDirectory *td) {
  LHoodEval->Write(td, Coefficients.data());
}

#ifdef USE_FHICL
FluxFitter::FluxFitterOptions
MakeFluxFitterOptions(fhicl::ParameterSet const &ps) {

  FluxFitter::FluxFitterOptions ffo;

  ffo.InpCoeffFile = ps.get<std::string>("InputCoefficientsFile", "");
  ffo.InpCoeffDir = ps.get<std::string>("InputCoefficientsDirectory", "");
  ffo.InputTargetFluxFile = ps.get<std::string>("InputTargetFluxFile", "");
  ffo.InputTargetFluxName = ps.get<std::string>("InputTargetFluxName", "");
  ffo.CoeffLimit = ps.get<double>("CoeffLimit", 10);
  ffo.MaxLikelihoodCalls = ps.get<size_t>("MaxLikelihoodCalls", 5E5);
  ffo.MINUITTolerance = ps.get<double>("MINUITTolerance", 1E-5);
  return ffo;
}
#endif
FluxFitter::FluxFitterOptions MakeFluxFitterOptions(int argc,
                                                    char const *argv[]) {

  FluxFitter::FluxFitterOptions ffo;

  ffo.IsGauss = false;
  ffo.InpCoeffFile = "";
  ffo.InpCoeffDir = "";
  ffo.InputTargetFluxFile = "";
  ffo.InputTargetFluxName = "";
  ffo.CoeffLimit = 10;
  ffo.MaxLikelihoodCalls = 5E5;
  ffo.MINUITTolerance = 1E-5;

  int opt = 1;
  while (opt < argc) {
    if (std::string(argv[opt]) == "-A") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      ffo.InpCoeffFile = params[0];
      if (params.size() > 1) {
        ffo.InpCoeffDir = params[1];
      } else {
        ffo.InpCoeffDir = "";
      }
    } else if (std::string(argv[opt]) == "-t") {
      std::vector<std::string> params =
          ParseToVect<std::string>(argv[++opt], ",");
      if (params.size() != 2) {
        std::cout << "[ERROR]: Recieved " << params.size()
                  << " entrys for -t, expected 2." << std::endl;
        throw;
      }
      ffo.InputTargetFluxFile = params[0];
      ffo.InputTargetFluxName = params[1];
    } else if (std::string(argv[opt]) == "-n") {
      ffo.MaxLikelihoodCalls = str2T<int>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-T") {
      ffo.MINUITTolerance = str2T<double>(argv[++opt]);
    } else if (std::string(argv[opt]) == "-c") {
      ffo.CoeffLimit = str2T<double>(argv[++opt]);
    }
    opt++;
  }
  return ffo;
}

std::string DumpFluxFitterOptions(FluxFitter::FluxFitterOptions const &ffo) {
  std::stringstream ss("");
  std::cout << "{" << std::endl;
  ss << "\tIsGauss: " << ffo.IsGauss << std::endl;
  ss << "\tInpCoeffFile: " << ffo.InpCoeffFile << std::endl;
  ss << "\tInpCoeffDir: " << ffo.InpCoeffDir << std::endl;
  ss << "\tInputTargetFluxFile: " << ffo.InputTargetFluxFile << std::endl;
  ss << "\tInputTargetFluxName: " << ffo.InputTargetFluxName << std::endl;
  ss << "\tCoeffLimit: " << ffo.CoeffLimit << std::endl;
  ss << "\tMaxLikelihoodCalls: " << ffo.MaxLikelihoodCalls << std::endl;
  ss << "\tMINUITTolerance: " << ffo.MINUITTolerance << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}
