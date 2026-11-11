#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

class LeadingAK15JetProducer : public edm::global::EDProducer<> {
public:
  explicit LeadingAK15JetProducer(const edm::ParameterSet& config)
      : srcToken_(consumes<pat::JetCollection>(config.getParameter<edm::InputTag>("src"))),
        sourceIndexName_(config.getParameter<std::string>("sourceIndexName")),
        sourceMultiplicityName_(config.getParameter<std::string>("sourceMultiplicityName")),
        sourceHTName_(config.getParameter<std::string>("sourceHTName")) {
    produces<pat::JetCollection>();
  }

  void produce(edm::StreamID,
               edm::Event& event,
               const edm::EventSetup&) const override {
    edm::Handle<pat::JetCollection> jets;
    event.getByToken(srcToken_, jets);

    auto output = std::make_unique<pat::JetCollection>();
    if (!jets->empty()) {
      double sourceHT = 0.0;
      for (const auto& sourceJet : *jets) {
        sourceHT += sourceJet.pt();
      }
      const auto leading = std::max_element(
          jets->begin(), jets->end(),
          [](const pat::Jet& lhs, const pat::Jet& rhs) { return lhs.pt() < rhs.pt(); });
      pat::Jet jet(*leading);
      jet.addUserInt(sourceIndexName_, static_cast<int>(std::distance(jets->begin(), leading)));
      jet.addUserInt(sourceMultiplicityName_, static_cast<int>(jets->size()));
      jet.addUserFloat(sourceHTName_, static_cast<float>(sourceHT));
      output->push_back(std::move(jet));
    }

    event.put(std::move(output));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.add<edm::InputTag>("src", edm::InputTag("packedPatJetsAK15PFCHSSoftDrop"));
    description.add<std::string>("sourceIndexName", "leadingAK15SourceJetIdx");
    description.add<std::string>("sourceMultiplicityName", "originalAK15Multiplicity");
    description.add<std::string>("sourceHTName", "originalAK15HT");
    descriptions.add("LeadingAK15JetProducer", description);
  }

private:
  edm::EDGetTokenT<pat::JetCollection> srcToken_;
  std::string sourceIndexName_;
  std::string sourceMultiplicityName_;
  std::string sourceHTName_;
};

DEFINE_FWK_MODULE(LeadingAK15JetProducer);

class LeadingAK15SubjetProducer : public edm::global::EDProducer<> {
public:
  explicit LeadingAK15SubjetProducer(const edm::ParameterSet& config)
      : srcToken_(consumes<pat::JetCollection>(config.getParameter<edm::InputTag>("src"))),
        sourceIndexName_(config.getParameter<std::string>("sourceIndexName")),
        parentIndexName_(config.getParameter<std::string>("parentIndexName")) {
    produces<pat::JetCollection>();
  }

  void produce(edm::StreamID,
               edm::Event& event,
               const edm::EventSetup&) const override {
    edm::Handle<pat::JetCollection> jets;
    event.getByToken(srcToken_, jets);

    auto output = std::make_unique<pat::JetCollection>();
    if (!jets->empty() && jets->front().nSubjetCollections() > 0) {
      for (const auto& source : jets->front().subjets(0)) {
        if (source.isNull()) {
          continue;
        }
        pat::Jet subjet(*source);
        subjet.addUserInt(sourceIndexName_, static_cast<int>(source.key()));
        subjet.addUserInt(parentIndexName_, 0);
        output->push_back(std::move(subjet));
      }
    }

    event.put(std::move(output));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.add<edm::InputTag>("src", edm::InputTag("leadingPatJetsAK15PFCHS"));
    description.add<std::string>("sourceIndexName", "leadingAK15SourceSubjetIdx");
    description.add<std::string>("parentIndexName", "parentJetIdx");
    descriptions.add("LeadingAK15SubjetProducer", description);
  }

private:
  edm::EDGetTokenT<pat::JetCollection> srcToken_;
  std::string sourceIndexName_;
  std::string parentIndexName_;
};

DEFINE_FWK_MODULE(LeadingAK15SubjetProducer);
