"""Actual LIME text explanations over frozen local pipelines, including calibration/pooling."""
import html
import re
import time
import numpy as np
import inference as engine
from dashboard import CATALOG

MAX_XAI_CHARS=5000
# Same deployment parity bound as test_unified; never accept a label change.
BATCH_PROBABILITY_ATOL=2e-4

def lime_probabilities(model_id,texts,deadline=None):
    """LIME may generate empty strings: evaluate them as zero-information model inputs.

    Public user input remains nonempty. Only LIME's internally generated neighborhoods
    bypass the user-input guard; the actual preprocessing/calibrator/pooling stay intact.
    """
    if model_id not in CATALOG:raise ValueError('Mô hình không nằm trong danh sách XAI.')
    spec=engine.registry()['models'][model_id]
    blocks=[]
    for start in range(0,len(texts),32):
        if deadline is not None and time.perf_counter()>deadline:
            raise TimeoutError('LIME vượt ngân sách 90 giây; hãy dùng văn bản ngắn hơn hoặc ít mẫu nhiễu hơn.')
        batch=texts[start:start+32]
        if any(not isinstance(t,str) or len(t)>MAX_XAI_CHARS or '\x00' in t for t in batch):
            raise ValueError('Văn bản nhiễu LIME không hợp lệ.')
        if spec['kind']=='base':
            p=engine.base_predict(spec['base'],batch,spec['calibrated'])['probability']
        else:
            member=[engine.base_predict(m,batch,True)['probability'] for m in spec['members']]
            p=engine.pool(np.stack(member,axis=1),spec['weights'])
        if p is None:raise ValueError('LIME cần probability hợp lệ; không giải thích raw margin như probability.')
        blocks.append(engine.probability_ok(p))
    return np.concatenate(blocks)

def highlight(text,weights):
    """Escape all untrusted text before adding signed, local-surrogate highlights."""
    scale=max([abs(w) for w in weights.values()]+[1e-12]);parts=[]
    for token in re.split(r'(\W+)',text):
        word=html.escape(token);weight=weights.get(token)
        if weight is None or abs(weight)<1e-12:parts.append(word);continue
        color='20,130,115' if weight>0 else '221,77,92'
        alpha=.08+.30*abs(weight)/scale
        parts.append(f'<span title="LIME P(Real): {weight:+.6f}" style="background:rgba({color},{alpha:.3f});padding:2px 1px;border-radius:3px">{word}</span>')
    return '<div style="line-height:2;white-space:pre-wrap;overflow-wrap:anywhere">'+''.join(parts)+'</div>'

def explain(model_id,text,num_samples=300,num_features=10):
    from lime.lime_text import LimeTextExplainer
    engine.validate_text(text)
    if len(text)>MAX_XAI_CHARS:raise ValueError('XAI giới hạn 5.000 ký tự để kiểm soát thời gian xử lý.')
    if num_samples not in [200,300,600]:raise ValueError('Số mẫu nhiễu không nằm trong ngân sách cho phép.')
    if not re.search(r'\w',text):raise ValueError('LIME cần ít nhất một từ hoặc ký tự chữ/số.')
    tick=time.perf_counter();deadline=tick+90
    original=engine.predict(model_id,[text])
    anchor=engine.probability_ok(original['probability'])[0]
    batch_anchor_max_difference=0.
    def neighborhood(texts):
        nonlocal batch_anchor_max_difference
        p=lime_probabilities(model_id,texts,deadline).copy()
        same=np.array([item==text for item in texts])
        if same.any():
            delta=float(np.max(np.abs(p[same]-anchor)))
            batch_anchor_max_difference=max(batch_anchor_max_difference,delta)
            if delta>BATCH_PROBABILITY_ATOL:
                raise ValueError(f'Xác suất LIME lệch pipeline dự đoán {delta:.6f}; dừng giải thích để kiểm tra.')
            if np.any(p[same].argmax(1)!=anchor.argmax()):
                raise ValueError('Nhãn thay đổi theo batch gần ngưỡng phân lớp; dừng giải thích để kiểm tra độ ổn định nhãn.')
            # FP16 kernels can vary with batch shape. Explain the exact singleton
            # prediction shown to the user, while keeping real perturbed outputs.
            p[same]=anchor
        return p
    exp=LimeTextExplainer(class_names=['Fake','Real'],random_state=42,split_expression=r'\W+',bow=True).explain_instance(
        text,neighborhood,labels=(1,),num_features=num_features,num_samples=num_samples)
    np.testing.assert_allclose(exp.predict_proba,original['probability'][0],atol=1e-7,rtol=0)
    weights=[{'token':str(t),'weight':float(w)} for t,w in exp.as_list(label=1)]
    if not all(np.isfinite(x['weight']) for x in weights):raise ValueError('Trọng số LIME không hữu hạn.')
    local=float(np.asarray(exp.local_pred).ravel()[0]);prob=original['probability'][0]
    return {'model_id':model_id,'label':str(original['labels'][0]),'prob_fake':float(prob[0]),'prob_real':float(prob[1]),
            'features':weights,'html':highlight(text,{x['token']:x['weight'] for x in weights}),
            'local_r2':float(exp.score),'local_prediction_real':local,'intercept_real':float(exp.intercept[1]),
            'absolute_local_error':abs(local-float(prob[1])),'num_samples':num_samples,'seed':42,
            'batch_anchor_max_difference':batch_anchor_max_difference,
            'batch_probability_atol':BATCH_PROBABILITY_ATOL,
            'truncated':bool(original['truncated'][0]),'seconds':time.perf_counter()-tick,
            'explained_class':'real','definition':'Positive coefficient increases local surrogate P(real); negative decreases it. Not causal.'}
