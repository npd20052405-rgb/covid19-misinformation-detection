"""CSV boundaries and proof that reference labels do not enter inference."""
import sys
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import batch_prediction as b


class FilePredictionChecks(unittest.TestCase):
    def test_tab_semicolon_and_trimmed_headers(self):
        for delimiter in ['\t',';']:
            payload=f' ID {delimiter} Text {delimiter}EXPECTED_LABEL\nR1{delimiter}"News, with a comma."{delimiter}Real\n'.encode('utf-8')
            with self.subTest(delimiter=repr(delimiter)):
                frame=b.parse_csv(payload)
                self.assertEqual(list(frame.columns),['id','text','expected_label'])
                self.assertEqual(frame.iloc[0].text,'News, with a comma.')
                self.assertEqual(frame.iloc[0].expected_label,'real')

    def test_utf8_quoted_text_and_optional_labels(self):
        frame=b.parse_csv('\ufefftext,expected_label\n"COVID-19, a virus-related disease.",Real\nAnother text,\n'.encode('utf-8'))
        self.assertEqual(list(frame.id),['1','2'])
        self.assertEqual(frame.iloc[0].text,'COVID-19, a virus-related disease.')
        self.assertEqual(list(frame.expected_label),['real',''])

    def test_reject_malformed_and_oversized_inputs(self):
        invalid=[b'',b'tweet\nnews',b'text,text\nx,y',b'text\n""',b'text\nx,y',
                 b'id,text\n1,x\n1,y',b'text,expected_label\nx,unknown',b'text\nx\x00',
                 b'text\n'+b'x'*(20001),b'x'*(b.MAX_BYTES+1),b'text\n'+b'x\n'*101]
        for payload in invalid:
            with self.subTest(size=len(payload)):
                with self.assertRaises(ValueError):b.parse_csv(payload)

    def test_labels_never_enter_model_and_batch_limit_is_respected(self):
        frame=pd.DataFrame({'id':[str(i) for i in range(70)],'text':[f'Example {i}' for i in range(70)],'expected_label':['fake']*70})
        calls=[]
        def predict(key,texts):
            calls.append((key,list(texts)))
            return dict(labels=np.array(['real']*len(texts)),probability=np.tile([.2,.8],(len(texts),1)),truncated=[False]*len(texts))
        with patch.object(b.engine,'registry',return_value={'models':{'known':{'label':'Known'}}}),patch.object(b.engine,'predict',side_effect=predict):
            rows,errors=b.predict_file(frame,['known'])
        self.assertFalse(errors)
        self.assertEqual([len(c[1]) for c in calls],[64,6])
        self.assertEqual(calls[0][1],frame.text.iloc[:64].tolist())
        self.assertTrue((rows.predicted_label=='real').all())
        self.assertFalse(rows.matches_reference.any())
        np.testing.assert_allclose(rows.prob_real,.8)

    def test_export_prevents_spreadsheet_formula_execution(self):
        frame=pd.DataFrame({'text':['=1+1','  @SUM(A1)','normal'],'prob_real':[.2,.3,.4]})
        data=b.export_csv(frame)
        self.assertTrue(data.startswith(b'\xef\xbb\xbf'))
        decoded=data.decode('utf-8-sig')
        self.assertIn("'=1+1",decoded)
        self.assertIn("'  @SUM(A1)",decoded)


class FileAppIntegration(unittest.TestCase):
    def test_uploaded_file_real_inference_and_download(self):
        from streamlit.testing.v1 import AppTest
        app_dir=Path(__file__).resolve().parent
        payload=b.DEMO.read_bytes()
        class Upload:
            def getvalue(self):return payload
        with patch('streamlit.file_uploader',return_value=Upload()):
            app=AppTest.from_file(str(app_dir/'app.py'),default_timeout=240).run()
            app.sidebar.radio[0].set_value('Dự đoán').run()
            next(w for w in app.radio if w.key=='prediction_input_mode').set_value('Tải file CSV').run()
            self.assertFalse(app.exception)
            app.multiselect[0].set_value(['roberta_cal','distilbert_cal','B3_learned_choquet'])
            next(w for w in app.button if w.key=='run_file_prediction').click().run(timeout=240)
            self.assertFalse(app.exception);self.assertFalse(app.error)
            saved=app.session_state['file_prediction_result']
            self.assertFalse(saved['errors'])
            rows=saved['rows'];self.assertEqual(len(rows),30)
            rb=rows[rows.model_id=='roberta_cal']
            self.assertTrue(rb.matches_reference.all())
            self.assertEqual((rb.predicted_label=='real').sum(),5)
            self.assertEqual((rb.predicted_label=='fake').sum(),5)
            self.assertEqual(len(app.get('download_button')),2)
            expected=pd.read_csv(app_dir.parents[1]/'reports/20261023_file_demo/candidate_predictions.csv').set_index(['model','id'])
            for r in rows.itertuples():
                ref=expected.loc[(r.model_id,r.id)]
                self.assertEqual(r.predicted_label,ref.predicted_label)
                np.testing.assert_allclose([r.prob_fake,r.prob_real],[ref.prob_fake,ref.prob_real],atol=2e-4,rtol=0)

    def test_invalid_upload_is_rejected_before_inference(self):
        from streamlit.testing.v1 import AppTest
        class Upload:
            def getvalue(self):return b'text\n""\n'
        with patch('streamlit.file_uploader',return_value=Upload()),patch.object(b.engine,'predict') as predict:
            app=AppTest.from_file(str(Path(__file__).resolve().parent/'app.py'),default_timeout=240).run()
            app.sidebar.radio[0].set_value('Dự đoán').run()
            next(w for w in app.radio if w.key=='prediction_input_mode').set_value('Tải file CSV').run()
            self.assertFalse(app.exception);self.assertTrue(app.error)
            predict.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
