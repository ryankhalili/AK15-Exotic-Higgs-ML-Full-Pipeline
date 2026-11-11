#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kMissing = -999.0;

bool hasBranch(TTree* tree, const char* name) {
  return tree != nullptr && tree->GetBranch(name) != nullptr;
}

std::string trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::string::size_type begin = value.find_first_not_of(whitespace);
  if (begin == std::string::npos) {
    return "";
  }
  const std::string::size_type end = value.find_last_not_of(whitespace);
  return value.substr(begin, end - begin + 1);
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int addInputs(TChain& chain, const char* inputSpec) {
  std::string spec = inputSpec ? inputSpec : "";
  if (spec.empty()) {
    return 0;
  }

  bool readList = false;
  if (spec[0] == '@') {
    spec = spec.substr(1);
    readList = true;
  } else if (endsWith(spec, ".txt") || endsWith(spec, ".list")) {
    readList = true;
  }

  if (!readList) {
    return chain.Add(spec.c_str());
  }

  std::ifstream inputs(spec.c_str());
  if (!inputs.is_open()) {
    std::cerr << "ERROR: could not open input list '" << spec << "'" << std::endl;
    return 0;
  }

  int added = 0;
  std::string line;
  while (std::getline(inputs, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    added += chain.Add(line.c_str());
  }
  return added;
}

void requireBranch(TTree* tree, const char* name) {
  if (!hasBranch(tree, name)) {
    throw std::runtime_error(std::string("Missing required branch: ") + name);
  }
}

template <typename T>
std::unique_ptr<TTreeReaderArray<T>> optionalArray(TTreeReader& reader, TTree* tree, const char* name) {
  if (!hasBranch(tree, name)) {
    return std::unique_ptr<TTreeReaderArray<T>>();
  }
  return std::unique_ptr<TTreeReaderArray<T>>(new TTreeReaderArray<T>(reader, name));
}

template <typename T>
std::unique_ptr<TTreeReaderValue<T>> optionalValue(TTreeReader& reader, TTree* tree, const char* name) {
  if (!hasBranch(tree, name)) {
    return std::unique_ptr<TTreeReaderValue<T>>();
  }
  return std::unique_ptr<TTreeReaderValue<T>>(new TTreeReaderValue<T>(reader, name));
}

double deltaPhiSigned(double phi1, double phi2) {
  double dphi = std::fmod(phi1 - phi2, 2.0 * M_PI);
  if (dphi > M_PI) {
    dphi -= 2.0 * M_PI;
  } else if (dphi <= -M_PI) {
    dphi += 2.0 * M_PI;
  }
  return dphi;
}

double deltaR(double eta1, double phi1, double eta2, double phi2) {
  const double deta = eta1 - eta2;
  const double dphi = deltaPhiSigned(phi1, phi2);
  return std::sqrt(deta * deta + dphi * dphi);
}

double readMuonIso(const std::string& requested,
                   const std::unique_ptr<TTreeReaderArray<Float_t>>& iso04All,
                   const std::unique_ptr<TTreeReaderArray<Float_t>>& iso03All,
                   const std::unique_ptr<TTreeReaderArray<Float_t>>& iso04Chg,
                   const std::unique_ptr<TTreeReaderArray<Float_t>>& iso03Chg,
                   unsigned int idx) {
  const std::string mode = lower(requested);
  if (mode == "muon_pfreliso04_all" && iso04All) return (*iso04All)[idx];
  if (mode == "muon_pfreliso03_all" && iso03All) return (*iso03All)[idx];
  if (mode == "muon_pfreliso04_chg" && iso04Chg) return (*iso04Chg)[idx];
  if (mode == "muon_pfreliso03_chg" && iso03Chg) return (*iso03Chg)[idx];

  if (mode == "auto" || mode.empty()) {
    if (iso04All) return (*iso04All)[idx];
    if (iso03All) return (*iso03All)[idx];
    if (iso04Chg) return (*iso04Chg)[idx];
    if (iso03Chg) return (*iso03Chg)[idx];
  }
  return std::numeric_limits<double>::infinity();
}

bool passesMuonId(const std::string& requested,
                  const std::unique_ptr<TTreeReaderArray<Bool_t>>& mediumId,
                  const std::unique_ptr<TTreeReaderArray<Bool_t>>& tightId,
                  unsigned int idx) {
  const std::string mode = lower(trim(requested));
  if (mode.empty() || mode == "none") return true;
  if (mode == "medium") return mediumId && (*mediumId)[idx];
  if (mode == "tight") return tightId && (*tightId)[idx];
  return false;
}

using LumiMask = std::map<UInt_t, std::vector<std::pair<UInt_t, UInt_t>>>;

LumiMask loadLumiMask(const char* pathArg) {
  LumiMask mask;
  const std::string path = trim(pathArg ? pathArg : "");
  if (path.empty()) {
    return mask;
  }

  std::ifstream input(path.c_str());
  if (!input.is_open()) {
    throw std::runtime_error("Could not open lumi-mask ranges file: " + path);
  }

  std::string line;
  unsigned int lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    const std::string::size_type comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream values(line);
    UInt_t run = 0;
    UInt_t first = 0;
    UInt_t last = 0;
    if (!(values >> run >> first >> last) || first == 0 || last < first) {
      throw std::runtime_error("Invalid lumi-mask range at line " + std::to_string(lineNumber));
    }
    mask[run].push_back(std::make_pair(first, last));
  }
  if (input.bad()) {
    throw std::runtime_error("Could not read lumi-mask ranges file: " + path);
  }
  return mask;
}

bool passesLumiMask(const LumiMask& mask, UInt_t run, UInt_t lumi) {
  if (mask.empty()) {
    return false;
  }
  const LumiMask::const_iterator found = mask.find(run);
  if (found == mask.end()) {
    return false;
  }
  for (const auto& range : found->second) {
    if (lumi >= range.first && lumi <= range.second) {
      return true;
    }
  }
  return false;
}

void labelCutflow(TH1D& hist, bool abcdMode) {
  hist.GetXaxis()->SetBinLabel(1, "processed");
  hist.GetXaxis()->SetBinLabel(2, "certified_lumi");
  hist.GetXaxis()->SetBinLabel(3, "hlt_pass");
  hist.GetXaxis()->SetBinLabel(4, "has_muon");
  hist.GetXaxis()->SetBinLabel(5, "muon_pass");
  hist.GetXaxis()->SetBinLabel(6, "has_ak15");
  hist.GetXaxis()->SetBinLabel(7, "ak15_pt_eta");
  hist.GetXaxis()->SetBinLabel(8, "muon_ak15_dphi");
  hist.GetXaxis()->SetBinLabel(9, "ht4_pass");
  hist.GetXaxis()->SetBinLabel(10, "met_pass");
  hist.GetXaxis()->SetBinLabel(11, abcdMode ? "abcd_partition" : "mt_pass");
  hist.GetXaxis()->SetBinLabel(12, "met_ak15_dphi");
  hist.GetXaxis()->SetBinLabel(13, "selected");
}

void labelAbcd(TH1D& hist) {
  hist.GetXaxis()->SetBinLabel(1, "A_iso_pass_mt_high");
  hist.GetXaxis()->SetBinLabel(2, "B_iso_pass_mt_low");
  hist.GetXaxis()->SetBinLabel(3, "C_iso_fail_mt_high");
  hist.GetXaxis()->SetBinLabel(4, "D_iso_fail_mt_low");
}

}  // namespace

int PhysicsAnalysisTreeProducer(const char* inputFile = "input_nano.root",
                                const char* outputFile = "physics_analysis.root",
                                const char* sampleLabelArg = "",
                                int isData = 0,
                                Long64_t maxEvents = -1,
                                double jetPtMin = 200.0,
                                double jetEtaMax = 3.0,
                                const char* leptonModeArg = "muon",
                                double muonPtMin = 30.0,
                                double muonEtaMax = 2.5,
                                double muonIsoMax = 0.3,
                                double minDeltaPhi = 1.5,
                                const char* muonIsoBranchArg = "auto",
                                const char* lumiMaskRangesArg = "",
                                const char* muonIdArg = "none",
                                double htJetPtMin = 30.0,
                                double htJetEtaMax = 2.4,
                                int htJetIdMin = 2,
                                double minEventHT = -1.0,
                                double minMetPt = -1.0,
                                double minMuonMetTransverseMass = -1.0,
                                double minMetJetDeltaPhi = -1.0,
                                const char* requiredHltBranchArg = "",
                                int abcdModeArg = 0,
                                double abcdIsoPassMax = 0.15,
                                double abcdIsoFailMax = 0.5,
                                double abcdMtBoundary = 50.0,
                                const char* metBranchPrefixArg = "MET") {
  const std::string leptonMode = lower(trim(leptonModeArg ? leptonModeArg : "muon"));
  if (leptonMode != "muon") {
    std::cerr << "ERROR: PhysicsAnalysisTreeProducer currently supports only lepton_mode=muon; got '"
              << leptonMode << "'" << std::endl;
    return 2;
  }
  const std::string muonIdMode = lower(trim(muonIdArg ? muonIdArg : "none"));
  const std::string requiredHltBranch =
      trim(requiredHltBranchArg ? requiredHltBranchArg : "");
  const std::string metBranchPrefix =
      trim(metBranchPrefixArg ? metBranchPrefixArg : "MET");
  const std::string metPtBranch = metBranchPrefix + "_pt";
  const std::string metPhiBranch = metBranchPrefix + "_phi";
  const bool abcdMode = abcdModeArg != 0;
  if (muonIdMode != "none" && muonIdMode != "medium" && muonIdMode != "tight") {
    std::cerr << "ERROR: muon_id must be none, medium, or tight; got '"
              << muonIdMode << "'" << std::endl;
    return 2;
  }
  if (abcdMode &&
      (abcdIsoPassMax <= 0.0 || abcdIsoFailMax <= abcdIsoPassMax || abcdMtBoundary < 0.0)) {
    std::cerr << "ERROR: invalid ABCD boundaries: require 0 < iso pass < iso fail and mT >= 0"
              << std::endl;
    return 2;
  }

  TChain chain("Events");
  const int added = addInputs(chain, inputFile);
  if (added == 0) {
    std::cerr << "ERROR: no input files matched '" << inputFile << "'" << std::endl;
    return 1;
  }

  TTree* tree = &chain;
  const char* requiredBranches[] = {
      "run",
      "luminosityBlock",
      "event",
      "nMuon",
      "Muon_pt",
      "Muon_eta",
      "Muon_phi",
      "nJet",
      "Jet_pt",
      "Jet_eta",
      "Jet_jetId",
      "nSuperFatJetAK15",
      "SuperFatJetAK15_pt",
      "SuperFatJetAK15_eta",
      "SuperFatJetAK15_phi",
      "SuperFatJetAK15_mass",
  };

  try {
    for (const char* branch : requiredBranches) {
      requireBranch(tree, branch);
    }
    requireBranch(tree, metPtBranch.c_str());
    requireBranch(tree, metPhiBranch.c_str());
  } catch (const std::exception& err) {
    std::cerr << "ERROR: " << err.what() << std::endl;
    return 3;
  }

  const Long64_t inputEntries = chain.GetEntries();
  if (inputEntries < 0) {
    std::cerr << "ERROR: could not determine input entry count" << std::endl;
    return 3;
  }

  TTreeReader reader(tree);
  TTreeReaderValue<UInt_t> run(reader, "run");
  TTreeReaderValue<UInt_t> luminosityBlock(reader, "luminosityBlock");
  TTreeReaderValue<ULong64_t> event(reader, "event");

  TTreeReaderValue<UInt_t> nMuon(reader, "nMuon");
  TTreeReaderArray<Float_t> muonPt(reader, "Muon_pt");
  TTreeReaderArray<Float_t> muonEta(reader, "Muon_eta");
  TTreeReaderArray<Float_t> muonPhi(reader, "Muon_phi");
  auto muonIso04All = optionalArray<Float_t>(reader, tree, "Muon_pfRelIso04_all");
  auto muonIso03All = optionalArray<Float_t>(reader, tree, "Muon_pfRelIso03_all");
  auto muonIso04Chg = optionalArray<Float_t>(reader, tree, "Muon_pfRelIso04_chg");
  auto muonIso03Chg = optionalArray<Float_t>(reader, tree, "Muon_pfRelIso03_chg");
  auto muonMediumId = optionalArray<Bool_t>(reader, tree, "Muon_mediumId");
  auto muonTightId = optionalArray<Bool_t>(reader, tree, "Muon_tightId");
  if (muonIdMode == "medium" && !muonMediumId) {
    std::cerr << "ERROR: medium muon ID requested but Muon_mediumId is missing" << std::endl;
    return 4;
  }
  if (muonIdMode == "tight" && !muonTightId) {
    std::cerr << "ERROR: tight muon ID requested but Muon_tightId is missing" << std::endl;
    return 4;
  }

  TTreeReaderValue<UInt_t> nRecoJet(reader, "nJet");
  TTreeReaderArray<Float_t> recoJetPt(reader, "Jet_pt");
  TTreeReaderArray<Float_t> recoJetEta(reader, "Jet_eta");
  TTreeReaderArray<Int_t> recoJetId(reader, "Jet_jetId");
  TTreeReaderValue<Float_t> metPt(reader, metPtBranch.c_str());
  TTreeReaderValue<Float_t> metPhi(reader, metPhiBranch.c_str());

  TTreeReaderValue<UInt_t> nAK15(reader, "nSuperFatJetAK15");
  TTreeReaderArray<Float_t> ak15Pt(reader, "SuperFatJetAK15_pt");
  TTreeReaderArray<Float_t> ak15Eta(reader, "SuperFatJetAK15_eta");
  TTreeReaderArray<Float_t> ak15Phi(reader, "SuperFatJetAK15_phi");
  TTreeReaderArray<Float_t> ak15Mass(reader, "SuperFatJetAK15_mass");
  auto jetSourceIdx =
      optionalArray<Int_t>(reader, tree, "SuperFatJetAK15_leadingAK15SourceJetIdx");
  auto jetOriginalMultiplicity =
      optionalArray<Int_t>(reader, tree, "SuperFatJetAK15_originalAK15Multiplicity");
  auto jetOriginalHT =
      optionalArray<Float_t>(reader, tree, "SuperFatJetAK15_originalAK15HT");
  auto genWeight = optionalValue<Float_t>(reader, tree, "genWeight");
  std::unique_ptr<TTreeReaderValue<Bool_t>> requiredHlt;
  if (!requiredHltBranch.empty()) {
    if (!hasBranch(tree, requiredHltBranch.c_str())) {
      std::cerr << "ERROR: required HLT branch is missing: " << requiredHltBranch << std::endl;
      return 4;
    }
    requiredHlt.reset(new TTreeReaderValue<Bool_t>(reader, requiredHltBranch.c_str()));
  }
  if (!isData && !genWeight) {
    std::cerr << "ERROR: MC input is missing required genWeight branch" << std::endl;
    return 4;
  }

  LumiMask lumiMask;
  try {
    lumiMask = loadLumiMask(lumiMaskRangesArg);
    if (isData && lumiMask.empty()) {
      throw std::runtime_error("Data requires a nonempty lumi mask");
    }
  } catch (const std::exception& err) {
    std::cerr << "ERROR: " << err.what() << std::endl;
    return 4;
  }

  TFile output(outputFile, "CREATE");
  if (output.IsZombie()) {
    std::cerr << "ERROR: could not create output file '" << outputFile << "'" << std::endl;
    return 5;
  }

  TTree out("Events", "Muon-channel AK15 analysis tree");
  TH1D cutflow("cutflow", "Muon-channel AK15 cutflow (raw events)", 13, 0.5, 13.5);
  TH1D cutflowWeighted(
      "cutflow_weighted", "Muon-channel AK15 cutflow (generator-weight sums)", 13, 0.5, 13.5);
  labelCutflow(cutflow, abcdMode);
  labelCutflow(cutflowWeighted, abcdMode);
  cutflow.Sumw2();
  cutflowWeighted.Sumw2();
  TH1D abcdYields("abcd_yields", "ABCD region raw events", 4, 0.5, 4.5);
  TH1D abcdYieldsWeighted(
      "abcd_yields_weighted", "ABCD region generator-weight sums", 4, 0.5, 4.5);
  labelAbcd(abcdYields);
  labelAbcd(abcdYieldsWeighted);
  abcdYields.Sumw2();
  abcdYieldsWeighted.Sumw2();
  TH1D ak4JetEtaPreselection(
      "ak4_jet_eta_preselection",
      "AK4 jet pseudorapidity before eta acceptance;AK4 jet #eta;sum of event weights",
      60,
      -5.0,
      5.0);
  TH1D ak4JetEtaPreselectionRaw(
      "ak4_jet_eta_preselection_raw",
      "AK4 jet pseudorapidity before eta acceptance;AK4 jet #eta;jets",
      60,
      -5.0,
      5.0);
  ak4JetEtaPreselection.Sumw2();
  ak4JetEtaPreselectionRaw.Sumw2();

  TH1D normalization("normalization", "Analysis normalization metadata", 5, 0.5, 5.5);
  normalization.GetXaxis()->SetBinLabel(1, "processed_events");
  normalization.GetXaxis()->SetBinLabel(2, "sum_genweights");
  normalization.GetXaxis()->SetBinLabel(3, "sum_abs_genweights");
  normalization.GetXaxis()->SetBinLabel(4, "selected_events");
  normalization.GetXaxis()->SetBinLabel(5, "selected_sum_genweights");

  std::string sample = (sampleLabelArg && sampleLabelArg[0] != '\0') ? sampleLabelArg : "sample";
  TTree analysisMetadata("AnalysisMetadata", "Muon-channel selection configuration");
  analysisMetadata.Branch("sampleLabel", &sample);
  Double_t metadataJetPtMin = jetPtMin;
  Double_t metadataJetEtaMax = jetEtaMax;
  Double_t metadataMuonPtMin = muonPtMin;
  Double_t metadataMuonEtaMax = muonEtaMax;
  Double_t metadataMuonIsoMax = muonIsoMax;
  Double_t metadataMinDeltaPhi = minDeltaPhi;
  Double_t metadataHtJetPtMin = htJetPtMin;
  Double_t metadataHtJetEtaMax = htJetEtaMax;
  Int_t metadataHtJetIdMin = htJetIdMin;
  Double_t metadataMinEventHT = minEventHT;
  Double_t metadataMinMetPt = minMetPt;
  Double_t metadataMinMuonMetTransverseMass = minMuonMetTransverseMass;
  Double_t metadataMinMetJetDeltaPhi = minMetJetDeltaPhi;
  Int_t metadataAbcdMode = abcdMode ? 1 : 0;
  Double_t metadataAbcdIsoPassMax = abcdIsoPassMax;
  Double_t metadataAbcdIsoFailMax = abcdIsoFailMax;
  Double_t metadataAbcdMtBoundary = abcdMtBoundary;
  std::string metadataMuonId = muonIdMode;
  std::string metadataRequiredHlt = requiredHltBranch;
  std::string metadataMetBranchPrefix = metBranchPrefix;
  std::string metadataMuonIsoBranch =
      lower(trim(muonIsoBranchArg ? muonIsoBranchArg : "auto"));
  analysisMetadata.Branch("jetPtMin", &metadataJetPtMin);
  analysisMetadata.Branch("jetEtaMax", &metadataJetEtaMax);
  analysisMetadata.Branch("muonPtMin", &metadataMuonPtMin);
  analysisMetadata.Branch("muonEtaMax", &metadataMuonEtaMax);
  analysisMetadata.Branch("muonIsoMax", &metadataMuonIsoMax);
  analysisMetadata.Branch("minDeltaPhi", &metadataMinDeltaPhi);
  analysisMetadata.Branch("htJetPtMin", &metadataHtJetPtMin);
  analysisMetadata.Branch("htJetEtaMax", &metadataHtJetEtaMax);
  analysisMetadata.Branch("htJetIdMin", &metadataHtJetIdMin);
  analysisMetadata.Branch("minEventHT", &metadataMinEventHT);
  analysisMetadata.Branch("minMetPt", &metadataMinMetPt);
  analysisMetadata.Branch(
      "minMuonMetTransverseMass", &metadataMinMuonMetTransverseMass);
  analysisMetadata.Branch("minMetJetDeltaPhi", &metadataMinMetJetDeltaPhi);
  analysisMetadata.Branch("requiredHlt", &metadataRequiredHlt);
  analysisMetadata.Branch("metBranchPrefix", &metadataMetBranchPrefix);
  analysisMetadata.Branch("abcdMode", &metadataAbcdMode);
  analysisMetadata.Branch("abcdIsoPassMax", &metadataAbcdIsoPassMax);
  analysisMetadata.Branch("abcdIsoFailMax", &metadataAbcdIsoFailMax);
  analysisMetadata.Branch("abcdMtBoundary", &metadataAbcdMtBoundary);
  analysisMetadata.Branch("muonId", &metadataMuonId);
  analysisMetadata.Branch("muonIsoBranch", &metadataMuonIsoBranch);
  analysisMetadata.Fill();

  TTree luminosityBlocks("LuminosityBlocks", "Certified luminosity blocks seen in data input");
  UInt_t lumiRun = 0;
  UInt_t lumiBlock = 0;
  luminosityBlocks.Branch("run", &lumiRun);
  luminosityBlocks.Branch("luminosityBlock", &lumiBlock);

  UInt_t outRun = 0;
  UInt_t outLuminosityBlock = 0;
  ULong64_t outEvent = 0;
  Long64_t inputEntry = -1;
  Int_t outIsData = isData ? 1 : 0;
  Int_t selectedLeptonPdgId = 13;
  Int_t selectedMuonIdx = -1;
  Int_t selectedMuonMediumId = -1;
  Int_t selectedMuonTightId = -1;
  Int_t selectedJetIdx = -1;
  Int_t selectedSourceJetIdx = -1;
  Int_t nInputMuon = 0;
  Int_t nInputAK15 = 0;
  Int_t nHTJet = 0;
  Int_t nHT15Jet = -1;
  Float_t eventHT = 0.0;
  Float_t eventHT4 = 0.0;
  Float_t eventHT15 = kMissing;
  Int_t eventHT15IsComplete = 0;
  Float_t outMetPt = kMissing;
  Float_t outMetPhi = kMissing;
  Float_t muonMetDeltaPhi = kMissing;
  Float_t muonMetTransverseMass = kMissing;
  Float_t selectedMuonPt = kMissing;
  Float_t metJetDeltaPhi = kMissing;
  Float_t selectedMuonEta = kMissing;
  Float_t selectedMuonPhi = kMissing;
  Float_t selectedMuonIso = kMissing;
  Float_t selectedLeptonPt = kMissing;
  Float_t selectedLeptonEta = kMissing;
  Float_t selectedLeptonPhi = kMissing;
  Float_t selectedLeptonIso = kMissing;
  Float_t jet_pt = kMissing;
  Float_t jet_eta = kMissing;
  Float_t jet_phi = kMissing;
  Float_t jet_mass = kMissing;
  Float_t muonJetDeltaR = kMissing;
  Float_t muonJetDeltaPhi = kMissing;
  Float_t muonJetSignedDeltaPhi = kMissing;
  Float_t outGenWeight = 1.0;
  Int_t hasGenWeight = genWeight ? 1 : 0;
  Int_t abcdRegion = 0;

  out.Branch("run", &outRun);
  out.Branch("luminosityBlock", &outLuminosityBlock);
  out.Branch("event", &outEvent);
  out.Branch("inputEntry", &inputEntry);
  out.Branch("sample", &sample);
  out.Branch("isData", &outIsData);
  out.Branch("nInputMuon", &nInputMuon);
  out.Branch("nInputAK15", &nInputAK15);
  out.Branch("selectedLeptonPdgId", &selectedLeptonPdgId);
  out.Branch("selectedLeptonPt", &selectedLeptonPt);
  out.Branch("selectedLeptonEta", &selectedLeptonEta);
  out.Branch("selectedLeptonPhi", &selectedLeptonPhi);
  out.Branch("selectedLeptonIso", &selectedLeptonIso);
  out.Branch("selectedMuonIdx", &selectedMuonIdx);
  out.Branch("selectedMuonMediumId", &selectedMuonMediumId);
  out.Branch("selectedMuonTightId", &selectedMuonTightId);
  out.Branch("selectedMuonPt", &selectedMuonPt);
  out.Branch("selectedMuonEta", &selectedMuonEta);
  out.Branch("selectedMuonPhi", &selectedMuonPhi);
  out.Branch("selectedMuonIso", &selectedMuonIso);
  out.Branch("selectedJetIdx", &selectedJetIdx);
  out.Branch("selectedSourceJetIdx", &selectedSourceJetIdx);
  out.Branch("nHTJet", &nHTJet);
  out.Branch("eventHT", &eventHT);
  out.Branch("eventHT4", &eventHT4);
  out.Branch("nHT15Jet", &nHT15Jet);
  out.Branch("eventHT15", &eventHT15);
  out.Branch("eventHT15IsComplete", &eventHT15IsComplete);
  out.Branch("met_pt", &outMetPt);
  out.Branch("met_phi", &outMetPhi);
  out.Branch("muonMetDeltaPhi", &muonMetDeltaPhi);
  out.Branch("muonMetTransverseMass", &muonMetTransverseMass);
  out.Branch("jet_pt", &jet_pt);
  out.Branch("metJetDeltaPhi", &metJetDeltaPhi);
  out.Branch("jet_eta", &jet_eta);
  out.Branch("jet_phi", &jet_phi);
  out.Branch("jet_mass", &jet_mass);
  out.Branch("muonJetDeltaR", &muonJetDeltaR);
  out.Branch("muonJetDeltaPhi", &muonJetDeltaPhi);
  out.Branch("muonJetSignedDeltaPhi", &muonJetSignedDeltaPhi);
  out.Branch("genWeight", &outGenWeight);
  out.Branch("hasGenWeight", &hasGenWeight);
  out.Branch("abcdRegion", &abcdRegion);

  TTree* abcdEvents = nullptr;
  if (abcdMode) {
    abcdEvents = out.CloneTree(0);
    abcdEvents->SetName("ABCDEvents");
    abcdEvents->SetTitle("Muon-channel AK15 ABCD candidate tree");
  }

  Long64_t processed = 0;
  Long64_t selected = 0;
  double sumGenWeights = 0.0;
  double sumAbsGenWeights = 0.0;
  double selectedSumGenWeights = 0.0;
  std::set<std::pair<UInt_t, UInt_t>> certifiedLumis;

  bool stoppedAtMaxEvents = false;
  while (true) {
    if (maxEvents >= 0 && processed >= maxEvents) {
      stoppedAtMaxEvents = true;
      break;
    }
    // Empty, schema-checked inputs need no reader advance; maxEvents=0 also reads nothing.
    if (inputEntries == 0 || !reader.Next()) {
      break;
    }
    ++processed;
    const double eventWeight = isData ? 1.0 : static_cast<double>(**genWeight);
    sumGenWeights += eventWeight;
    sumAbsGenWeights += std::abs(eventWeight);
    cutflow.Fill(1);
    cutflowWeighted.Fill(1, eventWeight);

    if (isData && !passesLumiMask(lumiMask, *run, *luminosityBlock)) {
      continue;
    }
    cutflow.Fill(2);
    cutflowWeighted.Fill(2, eventWeight);
    if (isData) {
      certifiedLumis.insert(std::make_pair(*run, *luminosityBlock));
    }

    if (requiredHlt && !**requiredHlt) {
      continue;
    }
    cutflow.Fill(3);
    cutflowWeighted.Fill(3, eventWeight);

    nInputMuon = static_cast<Int_t>(*nMuon);
    nInputAK15 = (jetOriginalMultiplicity && *nAK15 > 0)
                     ? (*jetOriginalMultiplicity)[0]
                     : static_cast<Int_t>(*nAK15);
    if (*nMuon > 0) {
      cutflow.Fill(4);
      cutflowWeighted.Fill(4, eventWeight);
    }
    int bestMuonIdx = -1;
    double bestMuonPt = -1.0;
    double bestMuonIso = std::numeric_limits<double>::infinity();
    for (UInt_t i = 0; i < *nMuon; ++i) {
      const double iso = readMuonIso(muonIsoBranchArg ? muonIsoBranchArg : "auto",
                                     muonIso04All,
                                     muonIso03All,
                                     muonIso04Chg,
                                     muonIso03Chg,
                                     i);
      if (muonPt[i] <= muonPtMin) continue;
      if (std::abs(muonEta[i]) >= muonEtaMax) continue;
      if (iso >= (abcdMode ? abcdIsoFailMax : muonIsoMax)) continue;
      if (!passesMuonId(muonIdMode, muonMediumId, muonTightId, i)) continue;
      if (muonPt[i] > bestMuonPt) {
        bestMuonPt = muonPt[i];
        bestMuonIdx = static_cast<int>(i);
        bestMuonIso = iso;
      }
    }

    if (bestMuonIdx < 0) {
      continue;
    }
    cutflow.Fill(5);
    cutflowWeighted.Fill(5, eventWeight);

    if (*nAK15 == 0) {
      continue;
    }
    cutflow.Fill(6);
    cutflowWeighted.Fill(6, eventWeight);

    int bestJetIdx = -1;
    for (UInt_t j = 0; j < *nAK15; ++j) {
      if (bestJetIdx < 0 || ak15Pt[j] > ak15Pt[bestJetIdx]) {
        bestJetIdx = static_cast<int>(j);
      }
    }
    if (bestJetIdx < 0 ||
        ak15Pt[bestJetIdx] <= jetPtMin ||
        std::abs(ak15Eta[bestJetIdx]) >= jetEtaMax) {
      continue;
    }
    cutflow.Fill(7);
    cutflowWeighted.Fill(7, eventWeight);

    const double bestSignedDeltaPhi =
        deltaPhiSigned(ak15Phi[bestJetIdx], muonPhi[bestMuonIdx]);
    if (std::abs(bestSignedDeltaPhi) <= minDeltaPhi) {
      continue;
    }
    cutflow.Fill(8);
    cutflowWeighted.Fill(8, eventWeight);
    for (UInt_t j = 0; j < *nRecoJet; ++j) {
      if (recoJetPt[j] <= htJetPtMin) continue;
      if (recoJetId[j] < htJetIdMin) continue;
      ak4JetEtaPreselection.Fill(recoJetEta[j], eventWeight);
      ak4JetEtaPreselectionRaw.Fill(recoJetEta[j]);
    }

    outRun = *run;
    outLuminosityBlock = *luminosityBlock;
    outEvent = *event;
    inputEntry = reader.GetCurrentEntry();
    outIsData = isData ? 1 : 0;
    selectedLeptonPdgId = 13;
    selectedMuonIdx = bestMuonIdx;
    selectedMuonMediumId = muonMediumId ? ((*muonMediumId)[bestMuonIdx] ? 1 : 0) : -1;
    selectedMuonTightId = muonTightId ? ((*muonTightId)[bestMuonIdx] ? 1 : 0) : -1;
    selectedJetIdx = bestJetIdx;
    selectedSourceJetIdx = jetSourceIdx ? (*jetSourceIdx)[bestJetIdx] : bestJetIdx;
    selectedMuonPt = muonPt[bestMuonIdx];
    selectedMuonEta = muonEta[bestMuonIdx];
    selectedMuonPhi = muonPhi[bestMuonIdx];
    selectedMuonIso = bestMuonIso;
    selectedLeptonPt = selectedMuonPt;
    selectedLeptonEta = selectedMuonEta;
    selectedLeptonPhi = selectedMuonPhi;
    selectedLeptonIso = selectedMuonIso;
    nHTJet = 0;
    eventHT4 = 0.0;
    for (UInt_t j = 0; j < *nRecoJet; ++j) {
      if (recoJetPt[j] <= htJetPtMin) continue;
      if (std::abs(recoJetEta[j]) >= htJetEtaMax) continue;
      if (recoJetId[j] < htJetIdMin) continue;
      eventHT4 += recoJetPt[j];
      ++nHTJet;
    }
    eventHT = eventHT4;
    eventHT15 = kMissing;
    nHT15Jet = -1;
    eventHT15IsComplete = 0;
    if (jetOriginalHT && *nAK15 > 0) {
      eventHT15 = (*jetOriginalHT)[bestJetIdx];
      nHT15Jet = nInputAK15;
      eventHT15IsComplete = 1;
    } else if (!jetOriginalMultiplicity || nInputAK15 == static_cast<Int_t>(*nAK15)) {
      eventHT15 = 0.0;
      for (UInt_t j = 0; j < *nAK15; ++j) {
        eventHT15 += ak15Pt[j];
      }
      nHT15Jet = static_cast<Int_t>(*nAK15);
      eventHT15IsComplete = 1;
    }
    outMetPt = *metPt;
    outMetPhi = *metPhi;
    muonMetDeltaPhi = std::abs(deltaPhiSigned(selectedMuonPhi, outMetPhi));
    muonMetTransverseMass = std::sqrt(
        std::max(0.0, 2.0 * static_cast<double>(selectedMuonPt) *
                          static_cast<double>(outMetPt) *
                          (1.0 - std::cos(muonMetDeltaPhi))));
    jet_pt = ak15Pt[bestJetIdx];
    jet_eta = ak15Eta[bestJetIdx];
    jet_phi = ak15Phi[bestJetIdx];
    jet_mass = ak15Mass[bestJetIdx];
    muonJetSignedDeltaPhi = bestSignedDeltaPhi;
    muonJetDeltaPhi = std::abs(bestSignedDeltaPhi);
    muonJetDeltaR = deltaR(selectedMuonEta, selectedMuonPhi, jet_eta, jet_phi);
    hasGenWeight = genWeight ? 1 : 0;
    metJetDeltaPhi = std::abs(deltaPhiSigned(outMetPhi, jet_phi));
    outGenWeight = genWeight ? **genWeight : 1.0;

    if (minEventHT >= 0.0 && eventHT4 <= minEventHT) {
      continue;
    }
    cutflow.Fill(9);
    cutflowWeighted.Fill(9, eventWeight);

    if (minMetPt >= 0.0 && outMetPt <= minMetPt) {
      continue;
    }
    cutflow.Fill(10);
    cutflowWeighted.Fill(10, eventWeight);

    if (!abcdMode && minMuonMetTransverseMass >= 0.0 &&
        muonMetTransverseMass <= minMuonMetTransverseMass) {
      continue;
    }
    cutflow.Fill(11);
    cutflowWeighted.Fill(11, eventWeight);

    if (minMetJetDeltaPhi >= 0.0 && metJetDeltaPhi <= minMetJetDeltaPhi) {
      continue;
    }
    cutflow.Fill(12);
    cutflowWeighted.Fill(12, eventWeight);

    abcdRegion = 0;
    if (abcdMode) {
      const bool isoPass = selectedMuonIso < abcdIsoPassMax;
      const bool isoFail = selectedMuonIso >= abcdIsoPassMax &&
                           selectedMuonIso < abcdIsoFailMax;
      const bool mtHigh = muonMetTransverseMass > abcdMtBoundary;
      if (isoPass) {
        abcdRegion = mtHigh ? 1 : 2;
      } else if (isoFail) {
        abcdRegion = mtHigh ? 3 : 4;
      }
      if (abcdRegion == 0) {
        continue;
      }
      abcdYields.Fill(abcdRegion);
      abcdYieldsWeighted.Fill(abcdRegion, eventWeight);
      abcdEvents->Fill();
      if (abcdRegion != 1) {
        continue;
      }
    }

    out.Fill();
    cutflow.Fill(13);
    cutflowWeighted.Fill(13, eventWeight);
    ++selected;
    selectedSumGenWeights += eventWeight;
  }

  if (!stoppedAtMaxEvents && inputEntries != 0 &&
      reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
    std::cerr << "ERROR: abnormal TTreeReader termination, status="
              << static_cast<int>(reader.GetEntryStatus())
              << ", processed=" << processed << std::endl;
    return 6;
  }
  const Long64_t expectedProcessed =
      maxEvents >= 0 ? std::min(maxEvents, inputEntries) : inputEntries;
  if (processed != expectedProcessed) {
    std::cerr << "ERROR: processed " << processed << " entries, expected "
              << expectedProcessed << std::endl;
    return 6;
  }

  normalization.SetBinContent(1, static_cast<double>(processed));
  normalization.SetBinContent(2, sumGenWeights);
  normalization.SetBinContent(3, sumAbsGenWeights);
  normalization.SetBinContent(4, static_cast<double>(selected));
  normalization.SetBinContent(5, selectedSumGenWeights);

  for (const auto& runLumi : certifiedLumis) {
    lumiRun = runLumi.first;
    lumiBlock = runLumi.second;
    luminosityBlocks.Fill();
  }

  output.cd();
  out.Write();
  if (abcdEvents != nullptr) {
    abcdEvents->Write();
    abcdYields.Write();
    abcdYieldsWeighted.Write();
  }
  cutflow.Write();
  cutflowWeighted.Write();
  ak4JetEtaPreselection.Write();
  ak4JetEtaPreselectionRaw.Write();
  normalization.Write();
  analysisMetadata.Write();
  luminosityBlocks.Write();
  output.Close();

  std::cout << "PhysicsAnalysisTreeProducer summary\n"
            << "  input: " << inputFile << "\n"
            << "  output: " << outputFile << "\n"
            << "  sample: " << sample << "\n"
            << "  lepton mode: " << leptonMode << "\n"
            << "  jet pt min: " << jetPtMin << "\n"
            << "  muon pt min: " << muonPtMin << "\n"
            << "  muon iso max: " << muonIsoMax << "\n"
            << "  muon ID: " << muonIdMode << "\n"
            << "  required HLT: "
            << (requiredHltBranch.empty() ? "none" : requiredHltBranch) << "\n"
            << "  MET branches: " << metPtBranch << ", " << metPhiBranch << "\n"
            << "  ABCD mode: " << abcdMode << "\n"
            << "  HT jet pt min: " << htJetPtMin << "\n"
            << "  HT jet eta max: " << htJetEtaMax << "\n"
            << "  HT jet ID min: " << htJetIdMin << "\n"
            << "  processed events: " << processed << "\n"
            << "  stop reason: " << (stoppedAtMaxEvents ? "max-events" : "end-of-input") << "\n"
            << "  certified data lumis: " << certifiedLumis.size() << "\n"
            << "  sum gen weights: " << sumGenWeights << "\n"
            << "  selected events: " << selected << "\n"
            << "  selected sum gen weights: " << selectedSumGenWeights << std::endl;

  return 0;
}
