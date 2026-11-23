"""Three-page Vietnamese coursework dashboard; real metrics, inference and LIME."""
from pathlib import Path
import sys,json,time
import pandas as pd
import streamlit as st

APP=Path(__file__).resolve().parent
sys.path.insert(0,str(APP))
import inference as engine
import dashboard
from xai_lime import explain,MAX_XAI_CHARS

st.set_page_config(page_title='COVID-19 · AI Dashboard',page_icon='🧬',layout='wide')
st.markdown('''<style>
.block-container{max-width:1320px;padding:4rem 2.5rem 3rem}
[data-testid="stSidebar"]{background:#f2f4f8}
[data-testid="stMetric"]{background:#f5f7fb;border:1px solid #e8edf4;border-radius:12px;padding:15px}
.eyebrow{font-size:.78rem;font-weight:700;letter-spacing:.14em;color:#107f89;margin-bottom:.7rem}
[data-testid="stMetricValue"]{font-size:clamp(1.15rem,1.6vw,1.85rem)}
[data-testid="stMetricLabel"] p{white-space:normal;overflow:visible}
.subtitle{color:#617088;line-height:1.6;margin-bottom:1.8rem}
.result-note{border-left:3px solid #128b83;padding:10px 15px;background:#f2faf8;border-radius:4px}
div[data-testid="stForm"]{border:1px solid #e4e9f0;border-radius:12px;padding:22px}
</style>''',unsafe_allow_html=True)
st.sidebar.title('🧬 COVID-19 · AI Lab')
st.sidebar.caption('CONSTRAINT 2021 · Đề tài môn học')
page=st.sidebar.radio('Điều hướng',['Tổng quan','Dự đoán','Giải thích (LIME)'])
st.sidebar.divider()
test_done=(APP/'metrics_unified.csv').exists()
st.sidebar.caption('Một giao thức · Validation & Test đã xem' if test_done else 'Chưa có bảng cuối')
try:
    reg=engine.registry()
except (OSError,ValueError,json.JSONDecodeError):
    st.error('Không xác minh được danh mục checkpoint. Hãy kiểm tra artifact; app không tự tải model thay thế.');st.stop()
release_hash=engine.digest(APP/'model_registry.json')
if st.session_state.get('_release_hash')!=release_hash:
    for key in ['prediction_result','lime_result','prediction_models','xai_models','file_prediction_result','file_prediction_models']:
        st.session_state.pop(key,None)
    engine.load_base.cache_clear();engine.load_expanded.cache_clear();engine.load_gap.cache_clear();engine.checked_unified.cache_clear()
    st.session_state['_release_hash']=release_hash

def title(eyebrow,heading,subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>',unsafe_allow_html=True)
    st.title(heading)
    st.markdown(f'<div class="subtitle">{subtitle}</div>',unsafe_allow_html=True)

def available_models():
    choices,missing=dashboard.available()
    if missing:st.warning(f'{len(missing)} lựa chọn đã bị ẩn do thiếu artifact hoặc checksum không hợp lệ.')
    if not choices:st.error('Không có mô hình chính với checkpoint hợp lệ.');st.stop()
    return choices

def protocol_name(key):
    return 'Thống nhất · Train 3.792 / calibration C / Ensemble M'

if page=='Tổng quan':
    title('CONSTRAINT 2021','Phát hiện tin giả COVID-19',
          'Một giao thức thống nhất · Validation 2.139 · Test mirror 2.140 mẫu')
    frame=dashboard.overview()
    a,b,c,d=st.columns(4)
    a.metric('Mô hình',len(dashboard.CATALOG));a.caption(f"{(frame.kind!='Kết hợp').sum()} mô hình đơn · {(frame.kind=='Kết hợp').sum()} kết hợp")
    b.metric('Val F1 cao nhất',f'{frame.weighted_f1.max():.2%}');b.caption('Kết quả development đã quan sát')
    c.metric('Validation','2.139');c.caption('Loại một mẫu trùng đã xác minh')
    d.metric('Test','2.140');d.caption('Đánh giá lại tập Test đã được xem')
    st.divider();st.subheader('Kết quả Validation và Test')
    names={'display_name':'Mô hình','kind':'Loại','weighted_f1':'Val Weighted-F1 (%)','macro_f1':'Val Macro-F1 (%)','accuracy':'Val Accuracy (%)','test_weighted_f1':'Test Weighted-F1 (%)','test_macro_f1':'Test Macro-F1 (%)','test_accuracy':'Test Accuracy (%)'}
    view=frame[list(names)].rename(columns=names).copy()
    for col in view:
        if col.endswith('(%)'):view[col]*=100
    st.dataframe(view.style.format({col:'{:.4f}' for col in view if col.endswith('(%)')},na_rep='N/A'),hide_index=True,use_container_width=True)
    st.caption('Mọi mô hình thành viên fit cùng 3.792 mẫu; calibration trên C, Ensemble học trên M. Không trộn điểm từ checkpoint cũ hoặc giao thức khác.')
    with st.expander('Đầy đủ độ đo và biểu đồ'):
        metrics=['precision_weighted','recall_weighted','precision_macro','recall_macro','precision_fake','recall_fake','f1_fake','precision_real','recall_real','f1_real','balanced_accuracy','mcc','log_loss','brier_real_binary','ece_top_10','mce_top_10','ece_real_10']
        columns=['display_name']+metrics+['test_'+m for m in metrics]
        st.dataframe(frame[[c for c in columns if c in frame]],hide_index=True,use_container_width=True)
        st.caption('Độ đo bổ sung dùng thang 0–1. Precision/Recall ghi rõ weighted, macro hoặc từng lớp; Brier cho P(Real); ECE/MCE 10 bin.')
        chart=frame.set_index('display_name')[['weighted_f1','test_weighted_f1']].mul(100).rename(columns={'weighted_f1':'Validation','test_weighted_f1':'Test'})
        st.bar_chart(chart,stack=False,color=['#198d88','#ef8354'])
    with st.expander('Giao thức và giới hạn'):
        st.write('Train nội bộ 3.792 mẫu học mô hình; 702 mẫu nội bộ chọn epoch; C=642 hiệu chỉnh xác suất; M=642 học Ensemble. Assignment nhóm cố định, seed42. H không dùng để học hoặc chọn trong đợt này.')
        st.write('Neural chọn checkpoint có cả gap loss/F1 nội bộ dưới0,1; ML dùng cấu hình cố định, không epoch. Ngân sách kiến trúc khác nhau được công bố trong báo cáo. Một seed và gap nhỏ không chứng minh ổn định hoặc không overfit.')
        st.write('Validation và Test đã được xem trong các đợt trước. Tham số đợt này khóa trước đánh giá; kết quả là development/exploratory, không phải Test độc lập mới. Test từ mirror được duyệt, chưa có checksum chính thức chứng nhận tương đương tuyệt đối; giữ nguyên2.140 mẫu, có2 trường hợp trùng chuẩn hóa với Train đã biết.')
        st.write('T5/TinyLlama chưa có checkpoint hoàn thành, không hiển thị như mô hình đã chạy. Dummy không hiển thị. Đây là pipeline ứng dụng, không tuyên bố thuật toán mới.')
    st.caption('Minh họa phân loại văn bản, không phải hệ thống xác minh sự thật y khoa hoặc tư vấn y tế.')

elif page=='Dự đoán':
    title('DỰ ĐOÁN','Dự đoán văn bản và file CSV',
          'Nhập văn bản hoặc tải CSV → chọn mô hình → xem kết quả cùng lúc.')
    choices=available_models()
    mode=st.radio('Cách nhập dữ liệu',['Nhập văn bản','Tải file CSV'],horizontal=True,key='prediction_input_mode')
    if mode=='Tải file CSV':
        from batch_prediction import render_file_prediction
        render_file_prediction(choices)
    else:
        with st.form('multi_predict_form'):
            input_col,config_col=st.columns([2,1])
            text=input_col.text_area('Văn bản tiếng Anh',height=220,max_chars=engine.MAX_CHARS,
                 placeholder='Dán bài đăng hoặc đoạn tin tiếng Anh về COVID-19…',key='prediction_text')
            selected=config_col.multiselect('Chọn mô hình',list(choices),
                 default=[k for k in ['ctbert_url_emoji_cal','distilbert_cal','ctbert_distilbert_weighted_mean'] if k in choices],
                 format_func=choices.get,key='prediction_models')
            config_col.caption('Có thể chọn nhiều mô hình. Văn bản được xử lý cục bộ.')
            submit=st.form_submit_button('Xem kết quả',type='primary',use_container_width=True)
        if submit:
            try:
                engine.validate_text(text)
                if not selected:raise ValueError('Hãy chọn ít nhất một mô hình.')
                rows=[];details={};errors=[];tick=time.perf_counter()
                with st.spinner('Đang suy luận lần lượt trên các checkpoint đã chọn…'):
                    for key in selected:
                        try:
                            result=engine.predict(key,[text]);p=result['probability'][0]
                            rows.append({'Mô hình':choices[key],'Nhãn':result['labels'][0].upper(),
                                'P(Fake)':float(p[0]),'P(Real)':float(p[1]),'Độ tin cậy':float(max(p)),
                                'Thời gian (s)':result['seconds'],'Giao thức':protocol_name(key),
                                'Cắt token':bool(result['truncated'][0])})
                            details[key]=result
                        except (ValueError,OSError,KeyError,RuntimeError,AssertionError) as exc:errors.append(choices[key]+': '+str(exc))
                st.session_state['prediction_result']={'rows':rows,'details':details,'errors':errors,
                    'text':text,'seconds':time.perf_counter()-tick,'requested':len(selected)}
            except ValueError as exc:st.error(str(exc))
        saved=st.session_state.get('prediction_result')
        if saved:
            st.divider();st.subheader('Kết quả')
            result_frame=pd.DataFrame(saved['rows']).rename(columns={'Confidence':'Độ tin cậy'})
            if len(result_frame)==len(saved['details']):
                result_frame['Mô hình']=[dashboard.CATALOG[k][0] for k in saved['details']]
            a,b,c=st.columns(3);a.metric('Mô hình chạy thành công',f"{len(result_frame)}/{saved['requested']}")
            b.metric('Tin giả',int((result_frame.Nhãn=='FAKE').sum()) if len(result_frame) else 0)
            c.metric('Tin thật',int((result_frame.Nhãn=='REAL').sum()) if len(result_frame) else 0)
            st.caption(f"Fake = tin giả · Real = tin thật · Thời gian: {saved['seconds']:.2f}s")
            if text!=saved['text']:st.info('Văn bản đã thay đổi; bấm phân loại để cập nhật bảng. Bảng dưới vẫn thuộc lượt trước.')
            if len(result_frame):
                st.dataframe(result_frame[['Mô hình','Nhãn','P(Fake)','P(Real)','Độ tin cậy']].style.format({'P(Fake)':'{:.2%}','P(Real)':'{:.2%}',
                    'Độ tin cậy':'{:.2%}'}),hide_index=True,use_container_width=True)
                st.bar_chart(result_frame.set_index('Mô hình')[['P(Fake)','P(Real)']],stack=False,color=['#df5966','#168a7a'])
                with st.expander('Thời gian và cách huấn luyện'):
                    st.dataframe(result_frame[['Mô hình','Thời gian (s)','Giao thức','Cắt token']],hide_index=True,use_container_width=True)
                for key,result in saved['details'].items():
                    if result['members']:
                        with st.expander('Thành viên và phép kết hợp · '+choices.get(key,dashboard.CATALOG[key][0])):
                            st.dataframe(pd.DataFrame([{'Thành viên':dashboard.MEMBER_NAMES.get(m,m),'Nhãn':x['labels'][0],
                                'P(Fake)':x['probability'][0,0],'P(Real)':x['probability'][0,1]} for m,x in result['members'].items()]),hide_index=True,use_container_width=True)
                            st.write('Trọng số / Möbius mass thực:',result['weights'])
                if result_frame['Cắt token'].any():st.warning('Một số Transformer đã cắt đầu vào ở giới hạn token của checkpoint.')
            for error in saved['errors']:st.error(error)
            st.caption('Độ tin cậy là ước lượng của mô hình, không phải bằng chứng tin đúng. Nhiều mô hình có thể cùng sai.')

else:
    title('GIẢI THÍCH LIME','Từ nào ảnh hưởng đến dự đoán?',
          'Xem các từ nổi bật trong lời giải thích cục bộ của mô hình.')
    choices=available_models()
    st.caption('Theo xấp xỉ LIME: xanh nghiêng về Real (tin thật), đỏ nghiêng về Fake (tin giả). Không xác nhận nội dung đúng/sai.')
    if st.button('Dùng văn bản vừa dự đoán'):
        previous=st.session_state.get('prediction_result',{}).get('text','')
        if len(previous)>MAX_XAI_CHARS:st.warning('Đoạn trước dài hơn giới hạn XAI 5.000 ký tự; hãy chọn đoạn ngắn hơn. Không tự cắt thay bạn.')
        else:st.session_state['xai_text']=previous
    with st.form('lime_form'):
        left,right=st.columns([2,1])
        text=left.text_area('Văn bản cần giải thích',height=180,max_chars=MAX_XAI_CHARS,key='xai_text',
             placeholder='Nhập cùng văn bản cần phân tích hoặc dùng nút lấy từ trang Dự đoán…')
        selected=right.multiselect('Mô hình cần giải thích',list(choices),default=['word_svm_cal'] if 'word_svm_cal' in choices else [],format_func=choices.get,key='xai_models')
        samples=right.select_slider('Số mẫu nhiễu LIME',options=[200,300,600],value=300)
        right.caption('300 mẫu là mức mặc định. Mô hình Transformer có thể chạy lâu hơn.')
        submitted=st.form_submit_button('Tạo giải thích LIME',type='primary',use_container_width=True)
    if submitted:
        try:
            engine.validate_text(text)
            if not selected:raise ValueError('Hãy chọn ít nhất một mô hình để giải thích.')
            explanations=[];errors=[]
            with st.spinner('Đang tạo nhiễu và tính LIME từ mô hình thật…'):
                for key in selected:
                    try:explanations.append(explain(key,text,num_samples=samples))
                    except ImportError:errors.append('Thiếu thư viện lime. Xem requirements.txt; không có trọng số thay thế giả.')
                    except (ValueError,OSError,KeyError,RuntimeError,AssertionError,TimeoutError) as exc:errors.append(choices[key]+': '+str(exc))
            st.session_state['lime_result']={'items':explanations,'errors':errors,'text':text}
        except ValueError as exc:st.error(str(exc))
    saved=st.session_state.get('lime_result')
    if saved:
        if text!=saved['text']:st.info('Đầu vào đã thay đổi; các giải thích dưới đây vẫn thuộc lần tính trước. Bấm tạo lại để cập nhật.')
        for error in saved['errors']:st.error(error)
        for result in saved['items']:
            with st.expander(dashboard.CATALOG[result['model_id']][0]+' → '+result['label'].upper(),expanded=True):
                a,b=st.columns(2)
                a.metric('P(Fake)',f"{result['prob_fake']:.2%}");b.metric('P(Real)',f"{result['prob_real']:.2%}")
                st.caption(f"Độ khớp giải thích (R²): {result['local_r2']:.2f} · {result['seconds']:.2f}s")
                if result['local_r2']<.5:st.warning('Mô hình giải thích khớp cục bộ thấp (R² < 0,5); không nên xem các từ nổi bật là lời giải thích đáng tin cậy.')
                if result['truncated']:st.warning('Checkpoint đã cắt văn bản theo giới hạn token; việc ẩn từ cũng có thể thay đổi các token được mô hình nhìn thấy.')
                st.markdown('**Từ nổi bật:** xanh tăng P(Real), đỏ giảm P(Real) trong xấp xỉ LIME.')
                features=result['features'][:5]
                if features:
                    for start in range(0,len(features),3):
                        for col,feature in zip(st.columns(min(3,len(features)-start)),features[start:start+3]):
                            col.metric(('🟢 ' if feature['weight']>0 else '🔴 ')+feature['token'],f"{feature['weight']:+.6f}")
                st.markdown(result['html'],unsafe_allow_html=True)
                if st.checkbox('Xem biểu đồ và thông số',key='lime_details_'+result['model_id']):
                    st.caption(f"{result['num_samples']} mẫu nhiễu · seed {result['seed']} · sai lệch tại đầu vào: {result['absolute_local_error']:.6f}. R² đo độ khớp giải thích, không phải accuracy.")
                    st.caption(f"Sai số xác suất do batch tại văn bản gốc: {result.get('batch_anchor_max_difference',0):.8f}. LIME giữ đúng xác suất dự đoán đơn; chặn nếu sai số > 0,0002 hoặc đổi nhãn.")
                    import matplotlib.pyplot as plt
                    weights=pd.DataFrame(result['features']).sort_values('weight')
                    fig,ax=plt.subplots(figsize=(9,max(2.5,.34*len(weights))))
                    ax.barh(weights.token,weights.weight,color=['#168a7a' if w>0 else '#df5966' for w in weights.weight])
                    ax.axvline(0,color='#64748b',lw=.8);ax.set(xlabel='Hệ số tuyến tính cục bộ cho P(Real)',ylabel='Từ được LIME chọn')
                    fig.tight_layout();st.pyplot(fig);plt.close(fig)
                    st.dataframe(weights,hide_index=True,use_container_width=True)
                    st.write('Intercept cho Real:',result['intercept_real'],'· Ước lượng tuyến tính tại đầu vào:',result['local_prediction_real'])
        st.caption('LIME giải thích cả dự đoán sai. Không có nhãn gốc nên không chấm đúng/sai cho văn bản nhập.')

with st.expander('Tên mô hình có nghĩa là gì?'):
    st.markdown("""- **SVM / Logistic / Naive Bayes:** ba cách phân loại văn bản khác nhau.
- **Từ / ký tự:** mô hình đọc đặc trưng theo từ hoặc cụm ký tự.
- **BERT-Tiny / DistilBERT:** hai mô hình Transformer xử lý ngôn ngữ.
- **Dấu +:** kết hợp dự đoán của hai mô hình được ghi tên.
- **Trọng số:** lấy trung bình xác suất có trọng số. **Choquet:** kết hợp bằng tích phân mờ.
- **Hiệu chỉnh xác suất:** điều chỉnh xác suất đầu ra; không bảo đảm dự đoán đúng. Tất cả checkpoint trong app học cùng Train 3.792 và được hiệu chỉnh trên C; Ensemble học phép kết hợp chỉ trên M.""")
