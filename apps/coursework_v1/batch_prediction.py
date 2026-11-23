"""Bounded CSV upload and real local batch inference; labels never enter models."""
import csv
import hashlib
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import inference as engine

MAX_BYTES=2*1024*1024
MAX_ROWS=100
MAX_TOTAL_CHARS=200000
DEMO=Path(__file__).resolve().parent/'demo_5_real_5_fake.csv'


def parse_csv(payload):
    if not isinstance(payload,bytes) or not payload:
        raise ValueError('File CSV rỗng hoặc không hợp lệ.')
    if len(payload)>MAX_BYTES:
        raise ValueError('File vượt giới hạn 2 MB.')
    try:
        text=payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Hãy lưu CSV bằng UTF-8 (CSV UTF-8 trong Excel).') from exc
    if '\x00' in text:raise ValueError('File chứa ký tự điều khiển không hợp lệ.')
    candidates=[]
    for delimiter in [',',';','\t']:
        try:
            fields=next(csv.reader(io.StringIO(text,newline=''),delimiter=delimiter,strict=True))
        except (csv.Error,StopIteration):continue
        if 'text' in [field.strip().lower() for field in fields]:
            candidates.append((delimiter,[field.strip().lower() for field in fields]))
    if not candidates:
        raise ValueError('File cần cột text; hỗ trợ dấu phẩy, chấm phẩy hoặc tab để phân cách cột.')
    if len(candidates)>1 and not all(fields==['text'] for _,fields in candidates):
        raise ValueError('Không xác định được dấu phân cách cột; hãy lưu lại CSV UTF-8 với một loại dấu phân cách.')
    reader=csv.reader(io.StringIO(text,newline=''),delimiter=candidates[0][0],strict=True)
    try:
        header=[field.strip().lower() for field in next(reader)]
        if len(header)!=len(set(header)) or any(not field for field in header):
            raise ValueError('CSV cần cột text; tên cột không được trùng nhau.')
        rows=[]
        for row in reader:
            if not row:continue
            if len(row)!=len(header):raise ValueError('CSV có dòng thiếu hoặc thừa cột; hãy kiểm tra dấu phẩy và dấu nháy.')
            rows.append(row)
            if len(rows)>MAX_ROWS:raise ValueError('Mỗi file tối đa 100 dòng văn bản.')
    except (csv.Error,StopIteration) as exc:
        raise ValueError('Không đọc được cấu trúc CSV.') from exc
    if not rows:raise ValueError('CSV chưa có dòng văn bản.')
    frame=pd.DataFrame(rows,columns=header)
    for index,value in enumerate(frame.text,1):
        try:engine.validate_text(value)
        except ValueError as exc:raise ValueError(f'Dòng dữ liệu {index}: {exc}') from exc
    if frame.text.str.len().sum()>MAX_TOTAL_CHARS:
        raise ValueError('Tổng văn bản vượt 200.000 ký tự; hãy chia thành file nhỏ hơn.')
    if 'id' not in frame:frame.insert(0,'id',[str(i) for i in range(1,len(frame)+1)])
    if frame.id.str.strip().eq('').any() or frame.id.str.len().gt(80).any() or not frame.id.is_unique:
        raise ValueError('Cột id phải duy nhất, không rỗng và không dài quá 80 ký tự.')
    if 'expected_label' in frame:
        frame['expected_label']=frame.expected_label.str.strip().str.lower()
        if not frame.expected_label.isin(['','fake','real']).all():
            raise ValueError('expected_label chỉ nhận Fake, Real hoặc để trống.')
    columns=[c for c in ['id','text','expected_label','source_url'] if c in frame]
    return frame[columns].copy()


def predict_file(frame,model_ids):
    reg=engine.registry()
    if not model_ids or len(set(model_ids))!=len(model_ids) or any(k not in reg['models'] for k in model_ids):
        raise ValueError('Hãy chọn ít nhất một mô hình hợp lệ, không trùng nhau.')
    if not 1<=len(frame)<=MAX_ROWS:raise ValueError('Số dòng không hợp lệ.')
    for value in frame.text:engine.validate_text(value)
    rows=[];errors=[]
    for model_id in model_ids:
        model_rows=[]
        try:
            for start in range(0,len(frame),64):
                chunk=frame.iloc[start:start+64]
                # Only the text column crosses the model boundary, never reference labels.
                result=engine.predict(model_id,chunk.text.tolist())
                probability=engine.probability_ok(result['probability'])
                labels=np.asarray(result['labels'])
                if len(probability)!=len(chunk) or not np.array_equal(labels,engine.CLASSES[probability.argmax(1)]):
                    raise ValueError('Nhãn hoặc số prediction không nhất quán.')
                for i,(_,source) in enumerate(chunk.iterrows()):
                    reference=source.get('expected_label','')
                    model_rows.append(dict(id=source.id,text=source.text,model_id=model_id,
                        model=reg['models'][model_id]['label'],predicted_label=str(labels[i]),
                        prob_fake=float(probability[i,0]),prob_real=float(probability[i,1]),
                        expected_label=reference,matches_reference=None if not reference else bool(labels[i]==reference),
                        truncated=bool(result['truncated'][i])))
            rows.extend(model_rows)
        except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as exc:
            errors.append(reg['models'][model_id]['label']+': '+str(exc))
    return pd.DataFrame(rows),errors


def export_csv(frame):
    safe=frame.copy()
    def literal(value):
        if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
            return "'"+value
        return value
    for column in safe.select_dtypes(include=['object','string']):
        safe[column]=safe[column].map(literal)
    return safe.to_csv(index=False).encode('utf-8-sig')


def render_file_prediction(choices):
    import streamlit as st
    st.caption('CSV UTF-8: tự nhận dấu phẩy, chấm phẩy hoặc tab. Cần cột text; tối đa 100 dòng, 2 MB.')
    if DEMO.is_file():
        st.download_button('Tải file minh họa: 5 Real + 5 Fake',DEMO.read_bytes(),
                           file_name=DEMO.name,mime='text/csv',key='download_demo_csv')
        st.caption('Bộ minh họa được chọn sau khi thử mô hình; không phải benchmark. Nhãn tham chiếu không được đưa vào mô hình.')
    upload=st.file_uploader('Chọn file CSV',type=['csv'],key='prediction_csv')
    if upload is None:return
    payload=upload.getvalue()
    try:frame=parse_csv(payload)
    except ValueError as exc:st.error(str(exc));return
    st.write(f'Đã đọc {len(frame)} đoạn văn bản.')
    with st.expander('Xem dữ liệu đầu vào'):
        st.dataframe(frame,hide_index=True,use_container_width=True)
    selected=st.multiselect('Chọn mô hình cho file',list(choices),
        default=['roberta_cal'] if 'roberta_cal' in choices else [],
        format_func=choices.get,key='file_prediction_models')
    signature=hashlib.sha256(payload).hexdigest()
    if st.button('Dự đoán toàn bộ file',type='primary',key='run_file_prediction'):
        if not selected:st.error('Hãy chọn ít nhất một mô hình.');return
        started=time.perf_counter()
        with st.spinner('Đang suy luận từng mô hình trên các dòng văn bản…'):
            result,errors=predict_file(frame,selected)
        st.session_state['file_prediction_result']=dict(signature=signature,models=selected,rows=result,
                                                        errors=errors,seconds=time.perf_counter()-started)
    saved=st.session_state.get('file_prediction_result')
    if not saved:return
    if saved['signature']!=signature or saved['models']!=selected:
        st.info('File hoặc danh sách mô hình đã thay đổi. Bấm dự đoán để cập nhật kết quả.');return
    for error in saved['errors']:st.error(error)
    result=saved['rows']
    if result.empty:return
    st.subheader('Kết quả toàn bộ file')
    pivot=result.pivot(index='id',columns='model',values='predicted_label').reindex(frame.id)
    pivot=pivot.apply(lambda col:col.str.upper())
    table=frame.set_index('id').join(pivot).reset_index()
    st.dataframe(table,hide_index=True,use_container_width=True)
    scored=result[result.expected_label!='']
    if not scored.empty:
        summary=scored.groupby('model',sort=False).agg(correct=('matches_reference','sum'),total=('id','size')).reset_index()
        summary['Đúng / tổng']=summary.correct.astype(int).astype(str)+' / '+summary.total.astype(str)
        st.dataframe(summary[['model','Đúng / tổng']].rename(columns={'model':'Mô hình'}),hide_index=True,use_container_width=True)
        st.caption('Đây chỉ là đối chiếu nhãn tham chiếu trong file, không thay thế đánh giá nghiên cứu độc lập.')
    with st.expander('Xem xác suất và chi tiết'):
        detail=result.rename(columns={'model':'Mô hình','predicted_label':'Dự đoán','prob_fake':'P(Fake)',
                                      'prob_real':'P(Real)','expected_label':'Nhãn tham chiếu','matches_reference':'Khớp tham chiếu'})
        st.dataframe(detail.style.format({'P(Fake)':'{:.2%}','P(Real)':'{:.2%}'}),hide_index=True,use_container_width=True)
    st.download_button('Tải kết quả dự đoán CSV',export_csv(result),file_name='ket_qua_du_doan.csv',mime='text/csv',key='download_file_predictions')
    st.caption(f'{len(result)} lượt dự đoán · {saved["seconds"]:.2f} giây · xử lý cục bộ, không gửi văn bản ra dịch vụ bên ngoài.')
    if result.truncated.any():st.warning('Một số văn bản đã được cắt theo giới hạn token của checkpoint.')
