import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_run2_nanoAOD_106Xv1_cff import run2_nanoAOD_106Xv1

process = cms.Process('NANO',Run2_2018,run2_nanoAOD_106Xv1)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register(
    'skipEvents',
    0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    'Number of input events to skip before processing'
)
options.register(
    'ak15LeadingOnly',
    False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    'Store custom AK15 tables only for the highest-pt AK15 jet in each event'
)
options.parseArguments()

process.maxEvents.input = cms.untracked.int32(options.maxEvents)

process.source = cms.Source("PoolSource",

    fileNames = cms.untracked.vstring(options.inputFiles),
    skipEvents = cms.untracked.uint32(max(0, options.skipEvents)),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)

process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('--python_filename nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

process.ak15ConstituentTable = cms.EDProducer(
    "AK15ConstituentTableProducer",
    jets = cms.InputTag("selectedPatJetsAK15PFCHS"),
    saveJetConstituents = cms.bool(True),
    debug = cms.bool(False),
    debugMaxJets = cms.uint32(5),
    jetTableName = cms.string("SuperFatJetAK15"),
    pfCandTableName = cms.string("SuperFatJetAK15PFCand"),
    genCandTableName = cms.string("SuperFatJetAK15GenCand"),
)

options.outputFile = options.outputFile.replace(".root", "_Nano.root")
process.NANOAODoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAOD'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = process.NANOAODEventContent.outputCommands
)

from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '106X_dataRun2_v35', '')

from JMEAnalysis.JetToolbox.jetToolbox_cff import jetToolbox

jetToolbox(
    process,
    'ak15',
    'ak15JetSubs',
    'out',
    PUMethod            = 'CHS',
    runOnMC              = False,
    addNsub=True,
    maxTau=4,
    addSoftDrop=True,
    addSoftDropSubjets=True,
    JETCorrPayload       = 'AK8PFchs',
    JETCorrLevels        = ['L1FastJet','L2Relative','L3Absolute','L2L3Residual'],
    bTagDiscriminators   = [
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probbb',
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probudsg',

    ],
)

from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
process.NjettinessAK15 = Njettiness.clone(
    src = cms.InputTag("selectedPatJetsAK15PFCHS"),
    Njets=cms.vuint32(1,2,3,4),
    R0 = cms.double( 1.5 )
)

process.leadingPatJetsAK15PFCHS = cms.EDProducer(
    "LeadingAK15JetProducer",
    src = cms.InputTag("packedPatJetsAK15PFCHSSoftDrop"),
    sourceIndexName = cms.string("leadingAK15SourceJetIdx"),
    sourceMultiplicityName = cms.string("originalAK15Multiplicity"),
    sourceHTName = cms.string("originalAK15HT"),
)

process.leadingPatSubjetsAK15PFCHS = cms.EDProducer(
    "LeadingAK15SubjetProducer",
    src = cms.InputTag("leadingPatJetsAK15PFCHS"),
    sourceIndexName = cms.string("leadingAK15SourceSubjetIdx"),
    parentIndexName = cms.string("parentJetIdx"),
)

from PhysicsTools.NanoAOD.common_cff import Var

process.jetAK15Table = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("selectedPatJetsAK15PFCHS"),
    cut = cms.string(""),
    name = cms.string("SuperFatJetAK15"),
    doc = cms.string("AK15 jets with Njettiness and SoftDrop mass"),
    singleton = cms.bool(False),
    subjets = cms.InputTag("ak15PFJetsCHSSoftDrop"),
    variables = cms.PSet(
        pt = Var("pt", float, doc="pt", precision=10),
        eta = Var("eta", float, doc="eta", precision=10),
        phi = Var("phi", float, doc="phi", precision=10),
        mass = Var("mass", float, doc="mass", precision=10),

        tau1 = Var("userFloat('NjettinessAK15CHS:tau1')",float,  doc="N-subjettiness tau1",precision=10),
        tau2 = Var("userFloat('NjettinessAK15CHS:tau2')", float, doc="N-subjettiness tau2", precision=10),
        tau3 = Var("userFloat('NjettinessAK15CHS:tau3')", float, doc="N-subjettiness tau3", precision=10),
        tau4 = Var("userFloat('NjettinessAK15CHS:tau4')", float, doc="N-subjettiness tau4", precision=10),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        nMuons = Var("?hasOverlaps('muons')?overlaps('muons').size():0", int, doc="number of muons in the jet"),
        muonIdx1 = Var("?overlaps('muons').size()>0?overlaps('muons')[0].key():-1", int, doc="index of first matching muon"),
        muonIdx2 = Var("?overlaps('muons').size()>1?overlaps('muons')[1].key():-1", int, doc="index of second matching muon"),
        electronIdx1 = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", int, doc="index of first matching electron"),
        electronIdx2 = Var("?overlaps('electrons').size()>1?overlaps('electrons')[1].key():-1", int, doc="index of second matching electron"),
        nElectrons = Var("?hasOverlaps('electrons')?overlaps('electrons').size():0", int, doc="number of electrons in the jet"),

 btag_pfDeepCSVJetTags_probb = Var("bDiscriminator('pfDeepCSVJetTags:probb')",float,doc="pfDeepCSVJetTags:probb",precision=10),
 btag_pfDeepCSVJetTags_probbb = Var("bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="pfDeepCSVJetTags:probbb",precision=10),
 btag_pfDeepCSVJetTags_probc = Var("bDiscriminator('pfDeepCSVJetTags:probc')",float,doc="pfDeepCSVJetTags:probc",precision=10),
 btag_pfDeepCSVJetTags_probudsg = Var("bDiscriminator('pfDeepCSVJetTags:probudsg')",float,doc="pfDeepCSVJetTags:probudsg",precision=10),

    )
)

process.genJetAK15Table = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("selectedPatJetsAK15PFCHS"),
    cut = cms.string("pt > 100."),
    name = cms.string("GenJetAK15"),
    doc  = cms.string("selectedPatJetsAK15PFCHS, i.e. ak15 Jets made with visible genparticles"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt  = Var("pt",  "float", doc="pt"),
        eta = Var("eta", "float", doc="eta"),
        phi = Var("phi", "float", doc="phi"),
        mass = Var("mass", "float", doc="mass"),

    )
)

from PhysicsTools.NanoAOD.common_cff import Var

process.subjetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("selectedPatJetsAK15PFCHSSoftDropPacked", "SubJets"),
    cut = cms.string(""),
    name = cms.string("SuperFat_SubJetAK8"),
    doc = cms.string("Softdrop subjets"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt  = Var("pt",  "float", doc="pt"),
        eta = Var("eta", "float", doc="eta"),
        phi = Var("phi", "float", doc="phi"),
        mass = Var("mass", "float", doc="mass"),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb') + bDiscriminator('pfDeepCSVJetTags:probbb')", "float", doc="DeepCSV b+bb discriminator"),
    )
)

if options.ak15LeadingOnly:
    leading_ak15_src = cms.InputTag("leadingPatJetsAK15PFCHS")
    process.jetAK15Table.src = leading_ak15_src
    process.jetAK15Table.variables.leadingAK15SourceJetIdx = Var(
        "userInt('leadingAK15SourceJetIdx')",
        int,
        doc="row index of the retained leading jet in selectedPatJetsAK15PFCHS",
    )
    process.jetAK15Table.variables.originalAK15Multiplicity = Var(
        "userInt('originalAK15Multiplicity')",
        int,
        doc="number of AK15 jets before leading-jet reduction",
    )
    process.jetAK15Table.variables.originalAK15HT = Var(
        "userFloat('originalAK15HT')",
        float,
        doc="scalar sum of reconstructed AK15 jet transverse momenta before leading-jet reduction",
        precision=10,
    )
    process.subjetTable.src = cms.InputTag("leadingPatSubjetsAK15PFCHS")
    process.subjetTable.variables.leadingAK15SourceSubjetIdx = Var(
        "userInt('leadingAK15SourceSubjetIdx')",
        int,
        doc="row index in the original AK15 soft-drop subjet collection",
    )
    process.subjetTable.variables.parentJetIdx = Var(
        "userInt('parentJetIdx')",
        int,
        doc="row index of the parent jet in SuperFatJetAK15",
    )
    process.genJetAK15Table.src = leading_ak15_src
    process.ak15ConstituentTable.jets = leading_ak15_src

process.nanoSequence += process.selectedPatJetsAK15PFCHS
process.nanoSequence += process.NjettinessAK15
if options.ak15LeadingOnly:
    process.nanoSequence += process.leadingPatJetsAK15PFCHS
    process.nanoSequence += process.leadingPatSubjetsAK15PFCHS
process.nanoSequence += process.subjetTable
process.nanoSequence += process.jetAK15Table

process.nanoSequence += process.ak15ConstituentTable

process.nanoAOD_step = cms.Path(process.nanoSequence)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

process.schedule = cms.Schedule(process.nanoAOD_step,process.endjob_step,process.NANOAODoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeData

process = nanoAOD_customizeData(process)

from Configuration.DataProcessing.Utils import addMonitoring

process = addMonitoring(process)

from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
