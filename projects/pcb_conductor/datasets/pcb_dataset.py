from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os.path as osp
from mmengine.fileio import get_file_backend

@DATASETS.register_module()
class PCBAConductorDataset(BaseSegDataset):
    METAINFO = dict(classes=('bg', 'fg'), palette=[(0, 0, 0), (255, 255, 255)])

    def __init__(self,
                 img_suffix='.bmp',
                 seg_map_suffix='.png',
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

@DATASETS.register_module()
class PCBBUnlabeledDataset(BaseSegDataset):
    METAINFO = dict(classes=('bg', 'fg'), palette=[(0, 0, 0), (255, 255, 255)])

    def __init__(self, img_suffix='.bmp', **kwargs):
        self.img_suffix = img_suffix
        super().__init__(img_suffix=img_suffix, **kwargs)

    def load_data_list(self):
        print('[DBG] PCBBUnlabeledDataset img_suffix =', self.img_suffix)
        img_dir = osp.normpath(self.data_prefix.get('img_path', ''))
        backend = get_file_backend(img_dir)

        # 1️⃣ 列出目录下所有文件（不在 backend 层做 suffix 过滤）
        all_files = list(backend.list_dir_or_file(
            img_dir, list_dir=False, recursive=True
        ))

        # 2️⃣ 用 Python 自己筛选 .bmp（大小写不敏感）
        file_list = [
            f for f in all_files
            if osp.splitext(f)[1].lower() == self.img_suffix.lower()
        ]

        if len(file_list) == 0:
            exts = sorted({osp.splitext(f)[1] for f in all_files})
            raise FileNotFoundError(
                f'No images found in {img_dir} with suffix {self.img_suffix}. '
                f'Found extensions: {exts}'
            )

        # 3️⃣ 生成 data_list
        data_list = [
            dict(img_path=osp.join(img_dir, rel_path))
            for rel_path in sorted(file_list)
        ]
        return data_list

