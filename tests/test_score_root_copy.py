"""Synthetic ROOT preservation for explicit single- and dual-model schemas."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_root_copy import augment, sha256, PF, KNOWN_SCORES, COMMON_BRANCHES, score_schema


def source(path,n=4):
    schema=dict(run='uint32',luminosityBlock='uint32',event='uint64',inputEntry='int64',
                source_id='string',event_key='string',abcdRegion='int32',nPart='int32',
                genWeight='float64',isData='bool',jet_pt='float32',jet_eta='float32',
                jet_phi='float32',jet_mass='float32')
    schema.update({'Part_'+suffix:'var * float32' for suffix in PF.values()})
    with uproot.create(path) as out:
        out.mktree('Events',schema)
        if n:
            batch={key:np.ones(n,dtype=dtype) for key,dtype in schema.items()
                   if dtype not in ['string','var * float32']}
            batch.update(inputEntry=np.arange(n,dtype='int64'),source_id=ak.Array(['source']*n),
                         event_key=ak.Array([str(i) for i in range(n)]),
                         abcdRegion=np.arange(n,dtype='int32')%4+1,
                         genWeight=np.array([(-1.)**i for i in range(n)]),
                         isData=np.arange(n)%2==0,nPart=np.full(n,2,dtype='int32'))
            batch.update({'Part_'+suffix:ak.values_astype(ak.Array([[1.,2.]]*n),np.float32)
                          for suffix in PF.values()})
            out['Events'].extend(batch)
        out['normalization']=(np.array([10.,8.]),np.arange(3,dtype=float))
        out['ProductionSpec']=json.dumps({'sample':'synthetic'})
        out.mktree('LuminosityBlocks',dict(run='uint32',luminosityBlock='uint32'))
        out['LuminosityBlocks'].extend(dict(run=np.array([1,1],dtype='uint32'),
                                           luminosityBlock=np.array([5,8],dtype='uint32')))
        out.mktree('AnalysisMetadata',dict(abcdMode='int32',muonEtaMax='float64'))
        out['AnalysisMetadata'].extend(dict(abcdMode=np.array([1],dtype='int32'),muonEtaMax=np.array([2.4])))


def predictor(names):
    def predict(batch):
        n=len(batch['event'])
        extra={name:np.full(n,.2 if name=='ParticleNet_score' else .8,dtype='float32') for name in names}
        extra.update(NN_valid=np.ones(n,dtype=bool),
                     NN_trainingRegion=ak.to_numpy(batch['abcdRegion'])==1,
                     NN_inputSplit=np.where(ak.to_numpy(batch['isData']),-1,-2).astype('int8'))
        return extra
    return predict


class RootCopyTests(unittest.TestCase):
    def test_single_and_both_models_preserve_original_objects_rows_and_signed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.root';source(path);original=sha256(path)
            for names in [('ParticleNet_score',),('ParticleTransformer_score',),KNOWN_SCORES]:
                with self.subTest(scores=names):
                    output=Path(temp)/('_'.join(names)+'.root')
                    result=augment(path,output,predictor(names),{},step=2,score_branches=names)
                    self.assertEqual(result['entries'],4)
                    self.assertEqual(set(result['score_sha256']),set(names))
                    self.assertEqual(sha256(path),original)
                    with uproot.open(path) as old,uproot.open(output) as new:
                        self.assertEqual(set(new['Events'].keys()),set(old['Events'].keys())|set(names)|set(COMMON_BRANCHES))
                        for absent in set(KNOWN_SCORES)-set(names):
                            self.assertNotIn(absent,new['Events'])
                        for key in old['Events'].keys():
                            self.assertTrue(ak.almost_equal(old['Events'][key].array(),new['Events'][key].array(),rtol=0,atol=0))
                        np.testing.assert_array_equal(new['Events']['genWeight'].array(library='np'),[1,-1,1,-1])
                        np.testing.assert_array_equal(new['Events']['NN_inputSplit'].array(library='np'),[-1,-2,-1,-2])
                        self.assertEqual(str(old['ProductionSpec']),str(new['ProductionSpec']))
                        self.assertEqual(new['LuminosityBlocks'].num_entries,2)
                        meta=json.loads(str(new['NNMetadata']))
                        self.assertEqual(meta['score_branches'],list(names))
                        self.assertEqual(meta['scored_models'],[name[:-len('_score')] for name in names])

    def test_empty_tree_has_only_selected_typed_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            path,output=Path(temp)/'in.root',Path(temp)/'out.root';source(path,0)
            augment(path,output,predictor(('ParticleNet_score',)),{},score_branches=('ParticleNet_score',))
            with uproot.open(output) as result:
                self.assertEqual(result['Events'].num_entries,0)
                self.assertEqual(result['Events']['ParticleNet_score'].typename,'float')
                self.assertNotIn('ParticleTransformer_score',result['Events'])

    def test_invalid_or_ambiguous_score_schema_rejected(self):
        for names in [(),('unknown_score',),('ParticleNet_score','ParticleNet_score'),'ParticleNet_score']:
            with self.subTest(schema=names):
                with self.assertRaises(ValueError):score_schema(names)
        with tempfile.TemporaryDirectory() as temp:
            path,output=Path(temp)/'in.root',Path(temp)/'out.root';source(path)
            with self.assertRaisesRegex(ValueError,'Checkpoint metadata'):
                augment(path,output,predictor(('ParticleNet_score',)),
                        {'checkpoint_sha256':{'ParticleTransformer':'fake'}},score_branches=('ParticleNet_score',))
            self.assertFalse(output.exists())

    def test_invalid_predictions_never_get_success_audit(self):
        names=('ParticleNet_score',)
        mutations={
            'extra_model':lambda values:values.update(ParticleTransformer_score=np.array([.1,.2],dtype='float32')),
            'missing_model':lambda values:values.pop('ParticleNet_score'),
            'nan_score':lambda values:values['ParticleNet_score'].__setitem__(0,np.nan),
            'invalid_split':lambda values:values['NN_inputSplit'].__setitem__(0,9),
            'float_split':lambda values:values.update(NN_inputSplit=np.array([-1.,-2.])),
            'boolean_split':lambda values:values.update(NN_inputSplit=np.array([True,False])),
            'wrong_data_split':lambda values:values['NN_inputSplit'].__setitem__(0,0),
            'wrong_region':lambda values:values['NN_trainingRegion'].__setitem__(0,False),
            'integer_flag':lambda values:values.update(NN_valid=np.ones(2,dtype='int32')),
        }
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'in.root';source(path,2);original=sha256(path)
            for label,mutate in mutations.items():
                with self.subTest(error=label):
                    output=Path(temp)/(label+'.root')
                    def bad(batch):
                        extra=predictor(names)(batch);mutate(extra);return extra
                    with self.assertRaises(ValueError):
                        augment(path,output,bad,{},step=2,score_branches=names)
                    self.assertFalse(output.with_suffix('.audit.json').exists())
                    self.assertEqual(sha256(path),original)

    def test_never_overwrites_input_output_or_existing_audit(self):
        names=('ParticleTransformer_score',)
        with tempfile.TemporaryDirectory() as temp:
            path,output=Path(temp)/'in.root',Path(temp)/'out.root';source(path)
            with self.assertRaises(ValueError):augment(path,path,predictor(names),{},score_branches=names)
            output.write_bytes(b'preserve')
            with self.assertRaises(ValueError):augment(path,output,predictor(names),{},score_branches=names)
            self.assertEqual(output.read_bytes(),b'preserve')
            other=Path(temp)/'other.root';other.with_suffix('.audit.json').write_text('preserve')
            with self.assertRaises(ValueError):augment(path,other,predictor(names),{},score_branches=names)
            self.assertFalse(other.exists())

    def test_rejects_any_previously_scored_input_and_unsupported_objects(self):
        names=('ParticleNet_score',)
        with tempfile.TemporaryDirectory() as temp:
            path,output,again=[Path(temp)/(name+'.root') for name in ['in','out','again']];source(path)
            augment(path,output,predictor(names),{},score_branches=names)
            with self.assertRaises(ValueError):
                augment(output,again,predictor(('ParticleTransformer_score',)),{},score_branches=('ParticleTransformer_score',))
            with uproot.update(path) as updated:
                updated['unsupported']=(np.ones((2,2)),np.arange(3,dtype=float),np.arange(3,dtype=float))
            with self.assertRaisesRegex(ValueError,'Unsupported ROOT object'):
                augment(path,again,predictor(names),{},score_branches=names)
            self.assertFalse(again.exists())


if __name__=='__main__':unittest.main()
