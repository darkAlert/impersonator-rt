import torch
from .models import BaseModel
from lwganrt.utils.detectors import PersonMaskRCNNDetector


class SegmentatorRT():
    def __init__(self, opt, device=None):
        self._name = 'SegmentatorRT'
        if device is not None:
            self.device = device
        else:
            assert torch.cuda.is_available()
            self.device = torch.device('cuda:' + str(opt['gpu_id']))

        self.detector = PersonMaskRCNNDetector(ks=opt['ks'], threshold=0.5, device=self.device,
                                               pretrained_path=opt['maskrcnn_path'])


    @torch.no_grad()
    def inference(self, src_img, apply_mask=False):
        src_img = src_img.to(self.device)

        _, ft_mask = self.detector.inference(src_img[0])

        if apply_mask:
            masked_img = (src_img + 1) / 2.0 * ft_mask
            masked_img = masked_img * 2 - 1.0
            masked_img = masked_img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        else:
            masked_img = None

        ft_mask = ft_mask.permute(0, 2, 3, 1)[0].cpu().detach().numpy()

        return ft_mask, masked_img


