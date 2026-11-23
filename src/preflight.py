"""Record actual local readiness without reading any dataset or Official Test."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = Path(__file__).resolve().parents[1]


def main():
    assert ROOT == EXPECTED.resolve(), 'Unexpected project root'
    now = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
    run_id = now.strftime('%Y%m%d_%H%M%S') + '_preflight_' + uuid.uuid4().hex[:8]
    for name in ['notebooks', 'data/raw/constraint2021', 'src', 'configs',
                 'models', 'results', 'figures', 'reports']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    dest = ROOT / 'results' / run_id
    dest.mkdir(exist_ok=False)
    versions = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'torch',
                 'transformers', 'nbformat', 'nbclient']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'cpu_count': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
           'versions': versions, 'ram_bytes': None}
    try:
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(n, ctypes.c_ulonglong) for n in ['total_phys', 'avail_phys',
                          'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended']]]
        mem = MemoryStatus()
        mem.length = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            env['ram_bytes'] = mem.total_phys
            env['available_ram_bytes'] = mem.avail_phys
    except (AttributeError, OSError) as exc:
        env['ram_probe_error'] = str(exc)
    try:
        import torch
        env['cuda_available'] = torch.cuda.is_available()
        env['gpu'] = torch.cuda.get_device_name(0) if env['cuda_available'] else None
        if env['cuda_available']:
            env['gpu_vram_bytes'] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        env['torch_probe_error'] = repr(exc)
    existing = ROOT / '00_Download_Dataset.ipynb'
    notebook = None
    if existing.exists():
        payload = existing.read_bytes()
        doc = json.loads(payload.decode('utf-8-sig'))
        import nbformat
        try:
            nbformat.validate(nbformat.from_dict(doc))
            valid = True
        except Exception as exc:
            valid = repr(exc)
        notebook = {'path': existing.name, 'sha256': hashlib.sha256(payload).hexdigest(),
                    'cells': len(doc.get('cells', [])), 'schema_valid': valid,
                    'nonempty_cells': sum(bool(''.join(c.get('source', [])).strip())
                                          for c in doc.get('cells', []))}
    sources = []
    for url in ['https://github.com/parthpatwa/covid19-fake-news-detection',
                'https://constraint-shared-task-2021.github.io/',
                'https://competitions.codalab.org/competitions/26655']:
        item = {'source_url': url, 'access_date': now.isoformat(),
                'download_method': 'urllib public page probe', 'archive_name': None,
                'sha256': None, 'license_or_terms': 'not verified'}
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                item.update(http_status=response.status, final_url=response.url)
        except Exception as exc:
            item['error'] = repr(exc)
        sources.append(item)
    state = {'run_id': run_id, 'date': now.isoformat(), 'project_root': str(ROOT),
             'official_test': 'LOCKED; no dataset content read',
             'stages': {'00': 'blocked_dataset_access', **{str(i).zfill(2): 'pending' for i in range(1, 7)}},
             'existing_notebook': notebook,
             'blockers': ['Official download/terms not verified in public session; CodaLab login required.',
                          'No local Train/Validation dataset available at initial inventory.'],
             'next_action': 'User supplies official Train/Validation files or authorizes a specific mirror.'}
    for filename, value in [('environment.json', env), ('source_probes.json', sources), ('state.json', state)]:
        with (dest / filename).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'results': str(dest), 'environment': env, 'state': state,
                      'sources': sources}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
