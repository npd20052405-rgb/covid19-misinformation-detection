"""Current unified app regression; inference parity uses M only, never tunes Val/Test."""
from pathlib import Path
import sys,json,unittest
import numpy as np
import pandas as pd
APP=Path(__file__).resolve().parent;ROOT=APP.parents[1]
sys.path.insert(0,str(APP));sys.path.insert(0,str(ROOT/'src'))
import inference as e
import dashboard as d
import unified_coursework as u

class UnifiedChecks(unittest.TestCase):
    def test_one_protocol_and_matching_rows(self):
        choices,missing=d.available();self.assertFalse(missing);self.assertEqual(len(choices),16)
        reg=e.registry();self.assertEqual(len(reg['bases']),12)
        self.assertEqual(len({x['protocol'] for x in reg['models'].values()}),1)
        f=d.overview();self.assertEqual(len(f),16);self.assertEqual(set(f.n),{2139});self.assertEqual(set(f.test_n),{2140})
        source=pd.concat([pd.read_csv(u.OUT/'final_metrics.csv'),pd.read_csv(APP/'metrics_ctbert.csv')],ignore_index=True)
        for _,row in f.iterrows():
            for split,prefix in [('Validation2139',''),('Test2140_reused','test_')]:
                r=source[(source.model==row.model)&(source.split==split)].iloc[0]
                self.assertAlmostEqual(row[prefix+'weighted_f1'],r.weighted_f1,places=12)
        self.assertFalse(any('dummy' in k for k in choices))

    def test_all_base_checkpoint_parity_on_M(self):
        m=u.old.partition('M').set_index('sample_id');evidence=[]
        for key in u.KEYS:
            pred=pd.read_csv(u.OUT/key/'M_calibrated_predictions.csv').iloc[:4]
            actual=e.predict(key+'_cal',m.loc[pred.sample_id,'tweet'].tolist())
            expected=pred[['prob_fake','prob_real']].to_numpy()
            delta=float(np.max(abs(actual['probability']-expected)))
            np.testing.assert_allclose(actual['probability'],expected,atol=2e-4,rtol=0)
            np.testing.assert_array_equal(actual['labels'],pred.predicted_label.to_numpy())
            empty=e.base_predict(key,[''],True)['probability'];self.assertTrue(np.isfinite(empty).all())
            evidence.append(dict(model=key,maximum_probability_difference=delta,labels_match=True))
            u.load.cache_clear()
        p=u.OUT/'app_parity.json'
        if not p.exists():u.save(p,evidence)

    def test_ctbert_release_inference_parity(self):
        import ctbert_tweet_preprocess as updated
        val=pd.read_csv(u.pm.RAW/'Constraint_Val.csv',dtype=str).set_index('id')
        cases=[('ctbert_url_emoji_cal', updated.OUT/'url_emoji_predictions.csv')]
        self.assertEqual([key for key in e.registry()['models'] if key.startswith('ctbert')], ['ctbert_url_emoji_cal'])
        for name,path in cases:
            pred=pd.read_csv(path).iloc[:4]
            ids=pred.sample_id.str.split(':').str[-1]
            actual=e.predict(name,val.loc[ids,'tweet'].tolist())
            np.testing.assert_allclose(actual['probability'],pred[['prob_fake','prob_real']],atol=2e-4,rtol=0)
            np.testing.assert_array_equal(actual['labels'],pred.predicted_label)
            updated.load.cache_clear()
            u.load.cache_clear()

    def test_ensemble_pool_and_input_guards(self):
        from unified_handover import MAIN_COMBINERS
        params=u.read(u.OUT/'combiner_parameters.json')
        for key in MAIN_COMBINERS:
            spec=params[key];frames=[pd.read_csv(u.OUT/m/'M_calibrated_predictions.csv').iloc[:4] for m in spec['members']]
            p=np.stack([f[['prob_fake','prob_real']].to_numpy() for f in frames],axis=1)
            np.testing.assert_allclose(e.pool(p,spec['weights']),u.pool(p,spec['weights']),atol=1e-14)
        for bad in ['', 'x'*20001, '\x00bad']:
            with self.assertRaises(ValueError):e.predict('word_svm_cal',[bad])

    def test_ui_and_lime(self):
        from streamlit.testing.v1 import AppTest
        from xai_lime import explain
        app=AppTest.from_file(str(APP/'app.py'),default_timeout=180).run()
        self.assertFalse(app.exception);self.assertEqual(len(app.selectbox),0)
        self.assertEqual(len(app.dataframe[0].value),16);self.assertEqual(app.metric[2].value,'2.139')
        app.sidebar.radio[0].set_value('Dự đoán').run()
        app.multiselect[0].set_value(['word_svm_cal','char_lr_cal','bert_tiny_cal'])
        text='The hospital reported new COVID-19 cases this week.'
        app.text_area[0].set_value(text);app.button[0].click().run()
        self.assertFalse(app.exception);self.assertEqual(len(app.dataframe[0].value),3)
        result=explain('word_svm_cal',text,num_samples=200)
        self.assertTrue(np.isfinite(result['local_r2']));self.assertTrue(result['features'])

if __name__=='__main__':unittest.main(verbosity=2)
