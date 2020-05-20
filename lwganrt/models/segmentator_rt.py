from .models import BaseModel
from lwganrt.utils.detectors import PersonMaskRCNNDetector


class SegmentatorRT(BaseModel):
    def __init__(self, opt, device):
        super(SegmentatorRT, self).__init__(opt)
        self._name = 'SegmentatorRT'
        self.device = device

        self.detector = PersonMaskRCNNDetector(ks=opt['ks'], threshold=0.5, device=self.device,
                                               pretrained_path=opt['maskrcnn_path'])

    @torch.no_grad()
    def inference(self, src_img):
        _, ft_mask = detector.inference(src_img)

        return ft_mask


def prepare_input(img, smpl, image_size=256, device=None):
    # resize image and convert the color space from [0, 255] to [-1, 1]
    if isinstance(img, np.ndarray):
        prep_img = cv_utils.transform_img(img, image_size, transpose=True) * 2 - 1.0
        prep_img = torch.tensor(prep_img, dtype=torch.float32).unsqueeze(0)
    else:
        raise NotImplementedError

    if isinstance(smpl, np.ndarray):
        if smpl.ndim == 1:
            prep_smpl = torch.tensor(smpl, dtype=torch.float32).unsqueeze(0)
        else:
            prep_smpl = torch.tensor(smpl, dtype=torch.float32)
    else:
        raise NotImplementedError

    if device is not None:
        prep_img = prep_img.to(device)
        prep_smpl = prep_smpl.to(device)

    return prep_img, prep_smpl


def apply_mask(img, mask):
    masked_img = (img + 1) / 2.0 * mask
    masked_img = masked_img * 2 - 1.0

    return masked_img



