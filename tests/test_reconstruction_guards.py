"""Local guard regressions; C++ checks are static, not ROOT/CMSSW validation.

The wrapper's actual embedded Python is executed against temporary files.
ROOT/CMSSW smoke tests (including empty trees and corrupt readers) remain needed.
Run independently of scientific dependencies: python -B tests/test_reconstruction_guards.py
"""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MACRO = (ROOT / 'ntuples/PhysicsAnalysisTreeProducer.C').read_text(encoding='utf-8')
WRAPPER = (ROOT / 'ntuples/run_physics_analysis_tree.sh').read_text(encoding='utf-8')
CONFIG = (ROOT / 'cmssw/nano_mc_cfg.py').read_text(encoding='utf-8')
MASK_CODE = WRAPPER.split('python - "${LUMI_MASK}" "${LUMI_MASK_RANGES}" <<\'PY\'\n', 1)[1].split('\nPY', 1)[0]


class MaskConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'mask.txt'
        self.output = self.directory / 'sidecar.txt'

    def convert(self, content=None, output=None):
        if content is not None:
            self.source.write_text(content, encoding='utf-8')
        return subprocess.run(
            [sys.executable, '-B', '-c', MASK_CODE, str(self.source), str(output or self.output)],
            capture_output=True, text=True, timeout=10,
        )

    def test_json_with_leading_whitespace(self):
        result = self.convert(' \n{"2": [[3, 5]], "1": [[1, 2]]}')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), '1 1 2\n2 3 5\n')

    def test_ranges_and_comments(self):
        result = self.convert('# certified\n1 2 3 # retained\n\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), '1 2 3\n')

    def test_empty_masks_fail_without_output(self):
        for content in ('', '{}', '{"1": []}', ' \n# empty\n'):
            with self.subTest(content=content):
                result = self.convert(content)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('nonempty lumi mask', result.stderr)
                self.assertFalse(self.output.exists())

    def test_invalid_masks_fail_without_output(self):
        for content in ('1 0 2', '1 4 2', '-1 1 2', '1 1 4294967296',
                        '1 1 2 extra', '{"1": [[1.5, 2]]}', '{"1": [[true, 2]]}'):
            with self.subTest(content=content):
                self.assertNotEqual(self.convert(content).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_existing_sidecar_preserved(self):
        self.output.write_bytes(b'preserve sidecar')
        self.assertNotEqual(self.convert('1 1 2').returncode, 0)
        self.assertEqual(self.output.read_bytes(), b'preserve sidecar')

    def test_source_output_alias_preserved(self):
        self.source.write_bytes(b'1 1 2\n')
        self.assertNotEqual(self.convert(output=self.source).returncode, 0)
        self.assertEqual(self.source.read_bytes(), b'1 1 2\n')

    def test_hardlink_alias_preserved(self):
        self.source.write_bytes(b'1 1 2\n')
        try:
            os.link(str(self.source), str(self.output))
        except OSError as error:
            self.skipTest('Hardlinks unavailable: %s' % error)
        self.assertNotEqual(self.convert().returncode, 0)
        self.assertEqual(self.source.read_bytes(), b'1 1 2\n')

    def test_dangling_symlink_preserved(self):
        target = self.directory / 'absent.txt'
        try:
            self.output.symlink_to(target)
        except OSError as error:
            self.skipTest('Symlinks unavailable: %s' % error)
        self.assertNotEqual(self.convert('1 1 2').returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_repeated_creation_fails(self):
        self.assertEqual(self.convert('1 1 2').returncode, 0)
        self.assertNotEqual(self.convert('2 3 4').returncode, 0)
        self.assertEqual(self.output.read_text(), '1 1 2\n')

    def test_exclusive_create_handles_precheck_race(self):
        self.source.write_text('1 1 2\n')
        self.output.write_bytes(b'concurrent writer')
        # Force the advisory precheck to miss an existing file; O_EXCL must still protect it.
        code = MASK_CODE.replace('if os.path.lexists(output):', 'if False:')
        result = subprocess.run(
            [sys.executable, '-B', '-c', code, str(self.source), str(self.output)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), b'concurrent writer')


class WrapperControlFlowTests(unittest.TestCase):
    def setUp(self):
        if os.name == 'nt':
            candidate = Path(os.environ.get('ProgramFiles', '')) / 'Git/bin/bash.exe'
            self.bash = str(candidate) if candidate.is_file() else None
        else:
            self.bash = shutil.which('bash')
        if not self.bash:
            self.skipTest('Local Bash unavailable; no WSL or remote fallback')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'input.root'
        self.source.write_bytes(b'synthetic placeholder; mock ROOT never opens it')
        self.config = self.directory / 'selection.json'
        self.config.write_text('{}')
        self.output = self.directory / 'output.root'

    def invoke(self, status):
        # Only exercise shell control flow; this is deliberately not a ROOT smoke test.
        command = 'root() { cat >/dev/null; return "$MOCK_ROOT_STATUS"; }; export -f root; bash "$@"'
        env = dict(os.environ, MOCK_ROOT_STATUS=str(status))
        return subprocess.run(
            [self.bash, '-c', command, '--',
             str(ROOT / 'ntuples/run_physics_analysis_tree.sh'), str(self.source), str(self.output),
             '--config', str(self.config)],
            env=env, capture_output=True, text=True, timeout=15,
        )

    def test_macro_failure_does_not_print_done(self):
        result = self.invoke(6)
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertNotIn('Done. Output file:', result.stdout)

    def test_success_control_flow(self):
        result = self.invoke(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Done. Output file:', result.stdout)

    def test_existing_output_refused_before_root(self):
        self.output.write_bytes(b'preserve ROOT output')
        result = self.invoke(0)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.output.read_bytes(), b'preserve ROOT output')


class ReconstructionStaticTests(unittest.TestCase):
    def test_root_writer_is_nonclobbering(self):
        self.assertIn('TFile output(outputFile, "CREATE");', MACRO)
        self.assertNotIn('"RECREATE"', MACRO)
        self.assertIn('[[ -e "${OUTPUT_FILE}" || -L "${OUTPUT_FILE}" ]]', WRAPPER)

    def test_empty_data_mask_rejected_before_output(self):
        guard = MACRO.index('if (isData && lumiMask.empty())')
        self.assertLess(guard, MACRO.index('TFile output('))
        predicate = MACRO.split('bool passesLumiMask(', 1)[1].split('void labelCutflow', 1)[0]
        self.assertIn('if (mask.empty()) {\n    return false;', predicate)
        self.assertIn('if (input.bad())', MACRO)

    def test_intentional_limit_checked_before_reader_advance(self):
        loop = MACRO.split('while (true) {', 1)[1]
        self.assertLess(loop.index('processed >= maxEvents'), loop.index('reader.Next()'))
        self.assertLess(loop.index('stoppedAtMaxEvents = true;'), loop.index('reader.Next()'))
        self.assertIn('if (inputEntries == 0 || !reader.Next())', loop)
        self.assertIn('stoppedAtMaxEvents ? "max-events" : "end-of-input"', MACRO)

    def test_abnormal_termination_and_short_reads_fail_before_publication(self):
        guard = MACRO.index('if (!stoppedAtMaxEvents && inputEntries != 0 &&')
        check = MACRO[guard:MACRO.index('normalization.SetBinContent(1,')]
        self.assertIn('reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd', check)
        self.assertIn('maxEvents >= 0 ? std::min(maxEvents, inputEntries) : inputEntries', check)
        self.assertIn('if (processed != expectedProcessed)', check)
        self.assertEqual(check.count('return 6;'), 2)
        self.assertLess(guard, MACRO.index('out.Write();'))
        self.assertNotRegex(check, r'selected\s*[<=>]')

    def test_sample_metadata_written_without_selected_rows(self):
        declaration = MACRO.index('std::string sample = (sampleLabelArg')
        metadata = MACRO.index('analysisMetadata.Branch("sampleLabel", &sample);')
        self.assertLess(declaration, metadata)
        self.assertLess(metadata, MACRO.index('analysisMetadata.Fill();'))
        self.assertLess(MACRO.index('analysisMetadata.Fill();'), MACRO.index('while (true)'))
        self.assertEqual(MACRO.count('std::string sample ='), 1)
        self.assertIn('out.Branch("sample", &sample);', MACRO)

    def test_wrapper_propagates_macro_exit_status(self):
        self.assertIn('set -euo pipefail', WRAPPER)
        self.assertIn('gROOT->LoadMacro("${ROOT_MACRO}+") < 0', WRAPPER)
        self.assertIn('int reductionStatus = 1;', WRAPPER)
        self.assertIn('reductionStatus = PhysicsAnalysisTreeProducer(', WRAPPER)
        self.assertLess(WRAPPER.index('gSystem->Exit(reductionStatus);'), WRAPPER.index('echo "Done.'))

    def test_misleading_table_removed_standard_nano_retained(self):
        ast.parse(CONFIG)
        self.assertNotIn('genJetAK15Table', CONFIG)
        self.assertNotIn('"GenJetAK15"', CONFIG)
        self.assertIn("process.load('PhysicsTools.NanoAOD.nano_cff')", CONFIG)
        self.assertIn('process.nanoAOD_step = cms.Path(process.nanoSequenceMC)', CONFIG)
        self.assertNotRegex(CONFIG, r'(?:remove|del).*genJetTable')


if __name__ == '__main__':
    unittest.main(verbosity=2)
