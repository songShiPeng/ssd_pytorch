import os
import Config
for root,dir,text_paths in os.walk(Config.dataset_root + '/core_500/Annotation'):
    for text_path in text_paths:
        if 'jy' in text_path or 'gs' in text_path:
            print('删除'+Config.dataset_root + '/core_500/Annotation/' + text_path)
            os.remove(Config.dataset_root + '/core_500/Annotation/' + text_path)
for root,dir,text_paths in os.walk(Config.dataset_root + '/core_500/Image'):
    for text_path in text_paths:
        if 'jy' in text_path or 'gs' in text_path:
            print('删除' + Config.dataset_root + '/core_500/Annotation/' + text_path)
            os.remove(Config.dataset_root + '/core_500/Image/'+text_path)