from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'apps/coursework_v1'
RESULTS = ROOT / 'results'
DEV = RESULTS / '20261110_ctbert_development_v1'
NEW = RESULTS / '20261110_ctbert_tweet_preprocess_v2'
TEST = RESULTS / '20261110_ctbert_test_mirror_v1'
NEW_TEST = RESULTS / '20261124_ctbert_url_emoji_test_v1'
REPORT = ROOT / 'reports/20261124_ctbert_coursework'
FIGURES = ROOT / 'figures/presentation'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def relative(path):
    return str(path.relative_to(ROOT))

def verify(path):
    for name, expected in read(path)['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Checksum mismatch: {name}')

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def publish_app():
    for folder in (DEV, NEW, TEST, NEW_TEST):
        verify(folder/'COMPLETE.json')
    verify(DEV/'development_lock.json')
    verify(NEW/'evaluation_lock.json')
    verify(DEV/'ctbert/base_lock.json')
    verify(NEW/'ctbert/base_lock.json')
    assert read(NEW_TEST/'protocol.json')['development_lock_sha256']==digest(NEW/'evaluation_lock.json')
    registry = read(APP/'model_registry.json')
    protocol = registry['models']['distilbert_cal']['protocol']
    lock = read(DEV/'development_lock.json')
    for key, folder, kind, label in [
        ('ctbert', DEV, 'ctbert_original', 'CT-BERT'),
        ('ctbert_url_emoji', NEW, 'ctbert_url_emoji', 'CT-BERT · URL và emoji'),
    ]:
        base_lock = folder/'ctbert/base_lock.json'
        cal = folder/'calibrator.joblib'
        files = dict(read(base_lock)['files'])
        files.update({relative(p):digest(p) for p in (base_lock, cal)})
        registry['bases'][key] = dict(kind=kind, calibrator=relative(cal), files=files)
        registry['models'][key+'_cal'] = dict(kind='base',label=label,base=key,
            metric_model=key,calibrated=True,protocol=protocol)
    for method, label in [('weighted_mean','Trung bình có trọng số'),('learned_choquet','Choquet')]:
        registry['models']['ctbert_distilbert_'+method] = dict(
            kind='ensemble',label=f'CT-BERT + DistilBERT · {label}',members=lock['members'],
            weights=lock['params'][method]['weights'],metric_model='ctbert_distilbert__'+method,
            protocol=protocol)
    names = ['ctbert','ctbert_distilbert__weighted_mean','ctbert_distilbert__learned_choquet']
    val = pd.read_csv(DEV/'metrics.csv').query('model in @names')
    added_val = pd.read_csv(NEW/'metrics.csv').query("model == 'url_emoji'").assign(model='ctbert_url_emoji')
    val = pd.concat([val,added_val]).assign(split='Validation2139')
    test = pd.concat([pd.read_csv(TEST/'metrics.csv'),pd.read_csv(NEW_TEST/'metrics.csv')]).assign(split='Test2140_reused')
    supplement = pd.concat([val,test],ignore_index=True).assign(protocol=protocol,representation='calibrated')
    assert len(supplement)==8 and not supplement.duplicated(['model','split']).any()
    supplement.to_csv(APP/'metrics_ctbert.csv',index=False)
    write_json(APP/'model_registry.json',registry)
    release=read(APP/'release.json')
    release.update(registry_sha256=digest(APP/'model_registry.json'),ctbert_metrics_sha256=digest(APP/'metrics_ctbert.csv'))
    write_json(APP/'release.json',release)
    return supplement


def make_figures(supplement):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    green,orange='#16877c','#d57140'
    paths=[]

    def save(fig,name):
        path=FIGURES/name
        fig.savefig(path,dpi=650,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        paths.append(path)

    fig,ax=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for row,(folder,name) in enumerate([(DEV,'CT-BERT'),(NEW,'CT-BERT with URL and emoji preprocessing')]):
        h=pd.read_csv(folder/'ctbert/history.csv')
        epoch=read(folder/'ctbert/base_lock.json')['selected']['epoch']
        for col,(metric,label) in enumerate([('loss','Cross-entropy loss'),('f1','Weighted F1')]):
            panel=ax[row,col]
            panel.plot(h.epoch,h['train_'+metric],'-o',c=green,label='Train (3,792)')
            panel.plot(h.epoch,h['dev_'+metric],'-o',c=orange,label='Inner validation (702)')
            panel.axvline(epoch,color='#4f596b',linestyle='--',label=f'Selected epoch {epoch}')
            panel.set(title=name,ylabel=label,xlabel='Epoch',xticks=h.epoch,ylim=(0,1))
            panel.legend(loc='center right',fontsize=8)
    save(fig,'06_train_dev_loss_f1_04_ctbert.png')

    labels=['CT-BERT','CT-BERT + DistilBERT (Weighted Mean)','CT-BERT + DistilBERT (Choquet)','CT-BERT (URL and emoji)']
    keys=['ctbert','ctbert_distilbert__weighted_mean','ctbert_distilbert__learned_choquet','ctbert_url_emoji']
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    for split,shift,color,label in [('Validation2139',-.15,green,'Validation (2,139)'),('Test2140_reused',.15,orange,'Test mirror (2,140)')]:
        values=supplement[supplement.split==split].set_index('model').loc[keys,'weighted_f1'].to_numpy()*100
        ax.scatter(values,np.arange(len(keys))+shift,c=color,label=label)
        for i,value in enumerate(values):
            ax.text(value+.025,i+shift,f'{value:.3f}',va='center',fontsize=9)
    ax.set(yticks=range(len(keys)),yticklabels=labels,xlim=(97.9,99),xlabel='Weighted F1 (%)',title='CT-BERT configurations on Validation and Test')
    ax.axvline(98.69,linestyle='--',color='#526479',label='Published Test reference (98.69%)')
    ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,fontsize=8)
    save(fig,'08_validation_va_test_ctbert.png')

    for split,prefix,title in [('Validation2139','09','Validation (2,139)'),('Test2140_reused','10','Test mirror (2,140)')]:
        fig,panels=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        fig.suptitle(f'CT-BERT confusion matrices | {title}')
        for panel,key,label in zip(panels.flat,keys,labels):
            r=supplement[(supplement.model==key)&(supplement.split==split)].iloc[0]
            matrix=np.array([[r.cm_fake_fake,r.cm_fake_real],[r.cm_real_fake,r.cm_real_real]],dtype=int)
            panel.imshow(matrix,cmap='Blues',vmin=0,vmax=1120)
            for i,j in np.ndindex(2,2):
                panel.text(j,i, str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>560 else '#163252')
            panel.set(title=label,xticks=[0,1],yticks=[0,1],xticklabels=['Fake','Real'],yticklabels=['Fake','Real'],xlabel='Predicted',ylabel='Reference')
        save(fig,f'{prefix}_confusion_ctbert.png')

    fig,panels=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for panel,split,title in zip(panels,['Validation2139','Test2140_reused'],['Validation (2,139)','Test mirror (2,140)']):
        panel.plot([0,1],[0,1],'--',c='#8593a3',label='Perfect calibration')
        dirs=[(DEV,DEV,DEV,NEW),(TEST,TEST,TEST,NEW_TEST)][split=='Test2140_reused']
        for folder,key,label,color in zip(dirs,keys,labels,[green,'#4769aa','#925ba0',orange]):
            bins=pd.read_csv(folder/((('url_emoji' if folder==NEW else key))+'_calibration_bins.csv'))
            bins=bins[(bins['mode']=='real_class')&(bins.n>0)]
            panel.plot(bins.mean_probability,bins.observed_frequency,'-o',c=color,label=label,markersize=3)
        panel.set(title=title,xlabel='Mean predicted P(Real)',ylabel='Observed Real proportion',xlim=(0,1),ylim=(0,1),aspect='equal')
    panels[1].legend(fontsize=7,loc='upper left')
    save(fig,'11_reliability_ctbert.png')
    return paths

def publish_tables(supplement):
    REPORT.mkdir(parents=True,exist_ok=True)
    combined=pd.concat([pd.read_csv(APP/'metrics_unified.csv'),supplement],ignore_index=True)
    assert not combined.duplicated(['model','split']).any()
    combined.to_csv(REPORT/'metrics_all_validation_test.csv',index=False)
    timing=RESULTS/'20261022_unified_coursework_v1/inference_timing.csv'
    import shutil
    shutil.copyfile(timing,REPORT/'timing_existing_models.csv')
    rows=[]
    for key,folder,test_folder in [('ctbert',DEV,TEST),('ctbert_url_emoji',NEW,NEW_TEST)]:
        chosen=read(folder/'ctbert/base_lock.json')
        validation=read(folder/'COMPLETE.json')
        test=read(test_folder/'COMPLETE.json')
        rows.append(dict(model=key,training_seconds=chosen['base_training_seconds'],peak_vram_bytes=chosen['peak_vram_bytes'],
            calibration_fit_seconds=read(DEV/'development_lock.json')['calibration_fit_seconds'] if folder==DEV else None,
            validation_inference_seconds=validation.get('inference_seconds',validation.get('base_inference_seconds')),
            test_inference_seconds=test['ctbert_inference_seconds'],
            **chosen['selected']))
    pd.DataFrame(rows).to_csv(REPORT/'ctbert_training_and_timing.csv',index=False)
    pd.DataFrame([dict(method=k,**v) for k,v in read(DEV/'development_lock.json')['params'].items()]).to_csv(REPORT/'ctbert_pooling_parameters.csv',index=False)
    paths=make_figures(supplement)
    write_json(REPORT/'manifest.json',{'files':{relative(p):digest(p) for p in [*paths,*REPORT.glob('*.csv'),APP/'metrics_ctbert.csv',APP/'model_registry.json',APP/'release.json']}})

if __name__=='__main__':
    publish_tables(publish_app())
