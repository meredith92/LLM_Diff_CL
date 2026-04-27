import copy
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import PackSegInputs
from mmcv.transforms import Compose


@TRANSFORMS.register_module()
class TwoViewAug(BaseTransform):
    def __init__(self, weak, strong):
        self.weak = Compose(weak)
        self.strong = Compose(strong)
        self.packer = PackSegInputs()

    def transform(self, results):
        rw = self.weak(copy.deepcopy(results))
        rs = self.strong(copy.deepcopy(results))

        bw = self.packer(rw)   # dict(inputs, data_samples)
        bs = self.packer(rs)

        ds = bs['data_samples']

        # ✅ 必须：把 weak view 存进去
        ds.set_field(bw['inputs'], 'inputs_w', field_type='data')

        # ✅ 必须：标记为 unlabeled（避免其它 transform 判错）
        ds.set_metainfo(dict(is_labeled=0, domain_id=1))

        # ✅ 返回只有 inputs + data_samples
        return dict(inputs=bs['inputs'], data_samples=ds)




@TRANSFORMS.register_module()
class EnsureLabeledUnlabeledKeys(BaseTransform):
    """Only set flags; NEVER create top-level inputs_w/inputs_s keys."""
    def transform(self, results):
        # Case 1: already packed (recommended). results has: inputs, data_samples
        if 'data_samples' in results and results['data_samples'] is not None:
            ds = results['data_samples']

            # If you already set is_labeled/domain_id earlier, keep them
            is_labeled = ds.metainfo.get('is_labeled', None)
            domain_id = ds.metainfo.get('domain_id', None)

            if is_labeled is None:
                # labeled samples will have gt_sem_seg (after PackSegInputs)
                is_labeled = 1 if getattr(ds, 'gt_sem_seg', None) is not None else 0
            if domain_id is None:
                domain_id = 0 if is_labeled else 1

            ds.set_metainfo(dict(is_labeled=int(is_labeled), domain_id=int(domain_id)))
            return results

        # Case 2: not packed yet (avoid creating inputs_w/inputs_s!)
        # Just add flags; PackSegInputs will ignore these unless you propagate later.
        if 'seg_map_path' in results or 'gt_seg_map' in results:
            results.setdefault('is_labeled', 1)
            results.setdefault('domain_id', 0)
        else:
            results.setdefault('is_labeled', 0)
            results.setdefault('domain_id', 1)

        return results

