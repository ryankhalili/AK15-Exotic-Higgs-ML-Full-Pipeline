#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class AK15ConstituentTableProducer : public edm::global::EDProducer<> {
public:
  explicit AK15ConstituentTableProducer(const edm::ParameterSet& iConfig);
  ~AK15ConstituentTableProducer() override = default;

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<pat::JetCollection> jetsToken_;

  bool saveJetConstituents_;
  bool debug_;
  unsigned int debugMaxJets_;
  std::string jetTableName_;
  std::string pfCandTableName_;
  std::string genCandTableName_;
};

AK15ConstituentTableProducer::AK15ConstituentTableProducer(const edm::ParameterSet& iConfig)
    : jetsToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      saveJetConstituents_(iConfig.getParameter<bool>("saveJetConstituents")),
      debug_(iConfig.existsAs<bool>("debug") ? iConfig.getParameter<bool>("debug") : false),
      debugMaxJets_(iConfig.existsAs<unsigned int>("debugMaxJets") ? iConfig.getParameter<unsigned int>("debugMaxJets") : 5),
      jetTableName_(iConfig.getParameter<std::string>("jetTableName")),
      pfCandTableName_(iConfig.getParameter<std::string>("pfCandTableName")),
      genCandTableName_(iConfig.getParameter<std::string>("genCandTableName")) {
  produces<nanoaod::FlatTable>("jetExt");
  produces<nanoaod::FlatTable>("pfCands");
  produces<nanoaod::FlatTable>("genCands");
}

void AK15ConstituentTableProducer::produce(edm::StreamID,
                                           edm::Event& iEvent,
                                           const edm::EventSetup&) const {
  edm::Handle<pat::JetCollection> jetsH;
  iEvent.getByToken(jetsToken_, jetsH);

  const auto& jets = *jetsH;

  std::vector<int> nPFConstituents(jets.size(), 0);
  std::vector<int> nGenConstituents(jets.size(), 0);
  std::vector<int> matchedGenJetExists(jets.size(), 0);
  std::vector<float> matchedGenJetPt(jets.size(), -999.f);
  std::vector<float> matchedGenJetEta(jets.size(), -999.f);
  std::vector<float> matchedGenJetPhi(jets.size(), -999.f);
  std::vector<float> matchedGenJetMass(jets.size(), -999.f);
  std::vector<int> hadronFlavor(jets.size(), 0);
  std::vector<int> partonFlavor(jets.size(), 0);

  std::vector<int> pf_jetIdx;
  std::vector<int> pf_candIdx;
  std::vector<int> pf_srcPackedCandIdx;
  std::vector<int> pf_pdgId;
  std::vector<int> pf_charge;
  std::vector<int> pf_fromPV;
  std::vector<int> pf_isElectron;
  std::vector<int> pf_isMuon;
  std::vector<int> pf_isPhoton;
  std::vector<int> pf_isChargedHadron;
  std::vector<int> pf_isNeutralHadron;

  std::vector<float> pf_pt;
  std::vector<float> pf_eta;
  std::vector<float> pf_phi;
  std::vector<float> pf_mass;
  std::vector<float> pf_energy;
  std::vector<float> pf_px;
  std::vector<float> pf_py;
  std::vector<float> pf_pz;
  std::vector<float> pf_puppiWeight;
  std::vector<float> pf_puppiWeightNoLep;
  std::vector<float> pf_dz;
  std::vector<float> pf_dxy;

  std::vector<int> gen_jetIdx;
  std::vector<int> gen_candIdx;
  std::vector<int> gen_srcCandIdx;
  std::vector<int> gen_pdgId;
  std::vector<int> gen_charge;
  std::vector<int> gen_status;

  std::vector<float> gen_pt;
  std::vector<float> gen_eta;
  std::vector<float> gen_phi;
  std::vector<float> gen_mass;
  std::vector<float> gen_energy;
  std::vector<float> gen_px;
  std::vector<float> gen_py;
  std::vector<float> gen_pz;

  for (size_t j = 0; j < jets.size(); ++j) {
    const auto& jet = jets[j];

    hadronFlavor[j] = jet.hadronFlavour();
    partonFlavor[j] = jet.partonFlavour();

    if (jet.genJet() != nullptr) {
      matchedGenJetExists[j] = 1;
      matchedGenJetPt[j] = jet.genJet()->pt();
      matchedGenJetEta[j] = jet.genJet()->eta();
      matchedGenJetPhi[j] = jet.genJet()->phi();
      matchedGenJetMass[j] = jet.genJet()->mass();
    }

    if (!saveJetConstituents_)
      continue;

    struct PFRow {
      edm::Ptr<reco::Candidate> ptr;
    };
    std::vector<PFRow> jetPFPtrs;

    for (unsigned int i = 0; i < jet.numberOfDaughters(); ++i) {
      auto cptr = jet.daughterPtr(i);
      if (cptr.isNull())
        continue;
      jetPFPtrs.push_back({cptr});
    }

    std::sort(jetPFPtrs.begin(), jetPFPtrs.end(),
              [](const PFRow& a, const PFRow& b) { return a.ptr->pt() > b.ptr->pt(); });

    int localIdx = 0;
    for (const auto& row : jetPFPtrs) {
      const reco::Candidate* cand = row.ptr.get();
      const auto* pc = dynamic_cast<const pat::PackedCandidate*>(cand);
      if (pc == nullptr)
        continue;

      pf_jetIdx.push_back(static_cast<int>(j));
      pf_candIdx.push_back(localIdx++);
      pf_srcPackedCandIdx.push_back(static_cast<int>(row.ptr.key()));
      pf_pdgId.push_back(pc->pdgId());
      pf_charge.push_back(pc->charge());
      pf_fromPV.push_back(static_cast<int>(pc->fromPV()));

      pf_pt.push_back(pc->pt());
      pf_eta.push_back(pc->eta());
      pf_phi.push_back(pc->phi());
      pf_mass.push_back(pc->mass());
      pf_energy.push_back(pc->energy());
      pf_px.push_back(pc->px());
      pf_py.push_back(pc->py());
      pf_pz.push_back(pc->pz());

      pf_puppiWeight.push_back(pc->puppiWeight());
      pf_puppiWeightNoLep.push_back(pc->puppiWeightNoLep());

      pf_dz.push_back(pc->bestTrack() ? pc->dz() : 0.f);
      pf_dxy.push_back(pc->bestTrack() ? pc->dxy() : 0.f);

      const int apid = std::abs(pc->pdgId());
      pf_isElectron.push_back(apid == 11 ? 1 : 0);
      pf_isMuon.push_back(apid == 13 ? 1 : 0);
      pf_isPhoton.push_back(apid == 22 ? 1 : 0);
      pf_isChargedHadron.push_back(apid == 211 ? 1 : 0);
      pf_isNeutralHadron.push_back(apid == 130 ? 1 : 0);
    }
    nPFConstituents[j] = localIdx;

    if (jet.genJet() != nullptr) {
      const auto* gj = jet.genJet();

      struct GenRow {
        edm::Ptr<reco::Candidate> ptr;
      };
      std::vector<GenRow> jetGenPtrs;

      for (unsigned int i = 0; i < gj->numberOfDaughters(); ++i) {
        auto gptr = gj->daughterPtr(i);
        if (gptr.isNull())
          continue;
        jetGenPtrs.push_back({gptr});
      }

      std::sort(jetGenPtrs.begin(), jetGenPtrs.end(),
                [](const GenRow& a, const GenRow& b) { return a.ptr->pt() > b.ptr->pt(); });

      int glocalIdx = 0;
      for (const auto& row : jetGenPtrs) {
        const reco::Candidate* gc = row.ptr.get();

        gen_jetIdx.push_back(static_cast<int>(j));
        gen_candIdx.push_back(glocalIdx++);
        gen_srcCandIdx.push_back(static_cast<int>(row.ptr.key()));
        gen_pdgId.push_back(gc->pdgId());
        gen_charge.push_back(gc->charge());

        gen_pt.push_back(gc->pt());
        gen_eta.push_back(gc->eta());
        gen_phi.push_back(gc->phi());
        gen_mass.push_back(gc->mass());
        gen_energy.push_back(gc->energy());
        gen_px.push_back(gc->px());
        gen_py.push_back(gc->py());
        gen_pz.push_back(gc->pz());

        const auto* gp = dynamic_cast<const reco::GenParticle*>(gc);
        if (gp != nullptr) {
          gen_status.push_back(gp->status());
        } else {
          gen_status.push_back(-1);
        }
      }
      nGenConstituents[j] = glocalIdx;
    }
  }

  if (debug_) {
    edm::LogVerbatim("AK15ConstituentTableProducer")
        << "event " << iEvent.id().run() << ":" << iEvent.id().luminosityBlock() << ":"
        << iEvent.id().event()
        << " ak15Jets=" << jets.size()
        << " savedPFConstituents=" << pf_pt.size()
        << " savedGenConstituents=" << gen_pt.size();

    const unsigned int maxJetsToPrint = std::min<unsigned int>(debugMaxJets_, jets.size());
    for (unsigned int j = 0; j < maxJetsToPrint; ++j) {
      const auto& jet = jets[j];
      edm::LogVerbatim("AK15ConstituentTableProducer")
          << "  jet " << j
          << " pt=" << jet.pt()
          << " eta=" << jet.eta()
          << " daughters=" << jet.numberOfDaughters()
          << " savedPF=" << nPFConstituents[j]
          << " savedGen=" << nGenConstituents[j]
          << " hasMatchedGenJet=" << matchedGenJetExists[j];
    }
  }

  auto jetExt = std::make_unique<nanoaod::FlatTable>(jets.size(), jetTableName_, false, true);
  jetExt->addColumn<int>("nPFConstituents", nPFConstituents,
                         "number of PF constituents stored for this AK15 jet",
                         nanoaod::FlatTable::IntColumn);
  jetExt->addColumn<int>("nGenConstituents", nGenConstituents,
                         "number of gen constituents stored for this AK15 jet",
                         nanoaod::FlatTable::IntColumn);
  jetExt->addColumn<int>("matchedGenJetExists", matchedGenJetExists,
                         "1 if this AK15 jet has a matched gen jet",
                         nanoaod::FlatTable::IntColumn);
  jetExt->addColumn<float>("matchedGenJetPt", matchedGenJetPt,
                           "matched gen-jet pt",
                           nanoaod::FlatTable::FloatColumn);
  jetExt->addColumn<float>("matchedGenJetEta", matchedGenJetEta,
                           "matched gen-jet eta",
                           nanoaod::FlatTable::FloatColumn);
  jetExt->addColumn<float>("matchedGenJetPhi", matchedGenJetPhi,
                           "matched gen-jet phi",
                           nanoaod::FlatTable::FloatColumn);
  jetExt->addColumn<float>("matchedGenJetMass", matchedGenJetMass,
                           "matched gen-jet mass",
                           nanoaod::FlatTable::FloatColumn);
  jetExt->addColumn<int>("hadronFlavor", hadronFlavor,
                         "hadron flavor from pat::Jet",
                         nanoaod::FlatTable::IntColumn);
  jetExt->addColumn<int>("partonFlavor", partonFlavor,
                         "parton flavor from pat::Jet",
                         nanoaod::FlatTable::IntColumn);

  auto pfTable = std::make_unique<nanoaod::FlatTable>(pf_pt.size(), pfCandTableName_, false, false);
  pfTable->addColumn<int>("jetIdx", pf_jetIdx,
                          "parent SuperFatJetAK15 row index",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("candIdx", pf_candIdx,
                          "local constituent index within parent jet after pt sorting",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("srcPackedCandIdx", pf_srcPackedCandIdx,
                          "original index in packedPFCandidates",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("pdgId", pf_pdgId,
                          "PDG id",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("charge", pf_charge,
                          "electric charge",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("fromPV", pf_fromPV,
                          "fromPV flag from pat::PackedCandidate",
                          nanoaod::FlatTable::IntColumn);

  pfTable->addColumn<float>("pt", pf_pt, "pt", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("eta", pf_eta, "eta", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("phi", pf_phi, "phi", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("mass", pf_mass, "mass", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("energy", pf_energy, "energy", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("px", pf_px, "px", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("py", pf_py, "py", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("pz", pf_pz, "pz", nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("puppiWeight", pf_puppiWeight,
                            "PUPPI weight",
                            nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("puppiWeightNoLep", pf_puppiWeightNoLep,
                            "PUPPI weight without lepton treatment",
                            nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("dz", pf_dz,
                            "dz if track exists, else 0",
                            nanoaod::FlatTable::FloatColumn);
  pfTable->addColumn<float>("dxy", pf_dxy,
                            "dxy if track exists, else 0",
                            nanoaod::FlatTable::FloatColumn);

  pfTable->addColumn<int>("isElectron", pf_isElectron,
                          "1 if abs(pdgId)==11",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("isMuon", pf_isMuon,
                          "1 if abs(pdgId)==13",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("isPhoton", pf_isPhoton,
                          "1 if abs(pdgId)==22",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("isChargedHadron", pf_isChargedHadron,
                          "1 if abs(pdgId)==211",
                          nanoaod::FlatTable::IntColumn);
  pfTable->addColumn<int>("isNeutralHadron", pf_isNeutralHadron,
                          "1 if abs(pdgId)==130",
                          nanoaod::FlatTable::IntColumn);

  auto genTable = std::make_unique<nanoaod::FlatTable>(gen_pt.size(), genCandTableName_, false, false);
  genTable->addColumn<int>("jetIdx", gen_jetIdx,
                           "parent SuperFatJetAK15 row index",
                           nanoaod::FlatTable::IntColumn);
  genTable->addColumn<int>("candIdx", gen_candIdx,
                           "local gen constituent index within matched gen jet after pt sorting",
                           nanoaod::FlatTable::IntColumn);
  genTable->addColumn<int>("srcGenCandIdx", gen_srcCandIdx,
                           "original daughterPtr key from matched gen jet",
                           nanoaod::FlatTable::IntColumn);
  genTable->addColumn<int>("pdgId", gen_pdgId,
                           "PDG id",
                           nanoaod::FlatTable::IntColumn);
  genTable->addColumn<int>("charge", gen_charge,
                           "electric charge",
                           nanoaod::FlatTable::IntColumn);
  genTable->addColumn<int>("status", gen_status,
                           "generator status if available, else -1",
                           nanoaod::FlatTable::IntColumn);

  genTable->addColumn<float>("pt", gen_pt, "pt", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("eta", gen_eta, "eta", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("phi", gen_phi, "phi", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("mass", gen_mass, "mass", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("energy", gen_energy, "energy", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("px", gen_px, "px", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("py", gen_py, "py", nanoaod::FlatTable::FloatColumn);
  genTable->addColumn<float>("pz", gen_pz, "pz", nanoaod::FlatTable::FloatColumn);

  iEvent.put(std::move(jetExt), "jetExt");
  iEvent.put(std::move(pfTable), "pfCands");
  iEvent.put(std::move(genTable), "genCands");
}

void AK15ConstituentTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("selectedPatJetsAK15PFCHS"));
  desc.add<bool>("saveJetConstituents", true);
  desc.add<bool>("debug", false);
  desc.add<unsigned int>("debugMaxJets", 5);
  desc.add<std::string>("jetTableName", "SuperFatJetAK15");
  desc.add<std::string>("pfCandTableName", "SuperFatJetAK15PFCand");
  desc.add<std::string>("genCandTableName", "SuperFatJetAK15GenCand");
  descriptions.add("AK15ConstituentTableProducer", desc);
}

DEFINE_FWK_MODULE(AK15ConstituentTableProducer);
