import numpy as np
import torch
import cv2
from skimage import transform as trans
from torchvision.transforms import v2, transforms
from math import floor, ceil

device = 'cuda'

DEFAULT_PARAMS = {
    'SwapperTypeTextSel': '128',
    'StrengthSwitch': False,
    'StrengthSlider': 100,
    'FaceAdjSwitch': False,
    'KPSXSlider': 0,
    'KPSYSlider': 0,
    'KPSScaleSlider': 0,
    'FaceScaleSlider': 0,
    'ColorSwitch': False,
    'ColorGammaSlider': 1.0,
    'ColorRedSlider': 0,
    'ColorGreenSlider': 0,
    'ColorBlueSlider': 0,
    'BorderTopSlider': 0,
    'BorderSidesSlider': 0,
    'BorderBottomSlider': 0,
    'BorderBlurSlider': 5,
    'BlendSlider': 5,
    'DiffSwitch': False,
    'DiffSlider': 10,
    'RestorerSwitch': False,
    'RestorerTypeTextSel': 'GFPGAN',
    'RestorerDetTypeTextSel': 'None',
    'RestorerSlider': 100,
    'OccluderSwitch': False,
    'OccluderSlider': 0,
    'FaceParserSwitch': False,
    'FaceParserSlider': 0,
    'MouthParserSlider': 0,
    'CLIPSwitch': False,
    'CLIPTextEntry': '',
    'CLIPSlider': 50,
}


class SwapCore:
    def __init__(self, models):
        self.models = models
        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.FFHQ_kps = np.array(
            [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
             [201.26117, 371.41043], [313.08905, 371.15118]])

    def get_embedding(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Detect face in image and return ArcFace embedding."""
        img = torch.from_numpy(image_rgb.astype('uint8')).to(device).permute(2, 0, 1)
        kpss = self.models.run_detect(img, 'Retinaface', max_num=1, score=0.5)
        if len(kpss) == 0:
            return None
        embedding, _ = self.models.run_recognize(img, kpss[0])
        return embedding

    def process(self, crop_rgb: np.ndarray, kps: np.ndarray,
                source_embedding: np.ndarray, parameters: dict = None) -> np.ndarray:
        """
        Swap the face in crop_rgb.

        crop_rgb : HxWx3 uint8 RGB numpy array (padded region around the face)
        kps      : 5x2 float32 keypoints in crop coordinate space
        source_embedding : 512-dim float32 ArcFace embedding of the source face
        returns  : HxWx3 uint8 RGB numpy array with the face replaced
        """
        if parameters is None:
            parameters = DEFAULT_PARAMS.copy()

        img = torch.from_numpy(crop_rgb.astype('uint8')).to(device).permute(2, 0, 1)

        img_x = img.size()[2]
        img_y = img.size()[1]
        scaled = False

        if img_x < 512 or img_y < 512:
            scaled = True
            if img_x <= img_y:
                new_h = int(512 * img_y / img_x)
                new_w = 512
            else:
                new_w = int(512 * img_x / img_y)
                new_h = 512
            tup = v2.Resize((new_h, new_w), antialias=True)
            img = tup(img)
            scale_x = new_w / img_x
            scale_y = new_h / img_y
            kps = kps * np.array([[scale_x, scale_y]], dtype=np.float32)

        img = self._swap_core(img, kps, source_embedding, parameters)

        if scaled:
            tdown = v2.Resize((img_y, img_x), antialias=True)
            img = tdown(img)

        return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def _swap_core(self, img, kps, s_e, parameters):
        """img: CxHxW CUDA uint8 tensor. Returns CxHxW CUDA tensor."""
        dst = self.arcface_dst * 4.0
        dst[:, 0] += 32.0

        if parameters['FaceAdjSwitch']:
            dst[:, 0] += parameters['KPSXSlider']
            dst[:, 1] += parameters['KPSYSlider']
            dst[:, 0] -= 255
            dst[:, 0] *= (1 + parameters['KPSScaleSlider'] / 100)
            dst[:, 0] += 255
            dst[:, 1] -= 255
            dst[:, 1] *= (1 + parameters['KPSScaleSlider'] / 100)
            dst[:, 1] += 255

        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)

        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        original_face_512 = v2.functional.affine(
            img, tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale, 0, center=(0, 0),
            interpolation=v2.InterpolationMode.BILINEAR)
        original_face_512 = v2.functional.crop(original_face_512, 0, 0, 512, 512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)

        latent = torch.from_numpy(self.models.calc_swapper_latent(s_e)).float().to(device)

        dim = 1
        if parameters['SwapperTypeTextSel'] == '128':
            dim = 1
            input_face_affined = original_face_128
        elif parameters['SwapperTypeTextSel'] == '256':
            dim = 2
            input_face_affined = original_face_256
        elif parameters['SwapperTypeTextSel'] == '512':
            dim = 4
            input_face_affined = original_face_512

        if parameters['FaceAdjSwitch']:
            input_face_affined = v2.functional.affine(
                input_face_affined, 0, (0, 0),
                1 + parameters['FaceScaleSlider'] / 100, 0,
                center=(dim * 128 - 1, dim * 128 - 1),
                interpolation=v2.InterpolationMode.BILINEAR)

        itex = 1
        if parameters['StrengthSwitch']:
            itex = ceil(parameters['StrengthSlider'] / 100.)

        output_size = int(128 * dim)
        output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=device)
        input_face_affined = input_face_affined.permute(1, 2, 0)
        input_face_affined = torch.div(input_face_affined, 255.0)

        for k in range(itex):
            for j in range(dim):
                for i in range(dim):
                    disc = input_face_affined[j::dim, i::dim]
                    disc = disc.permute(2, 0, 1)
                    disc = torch.unsqueeze(disc, 0).contiguous()
                    swapper_out = torch.empty((1, 3, 128, 128), dtype=torch.float32, device=device).contiguous()
                    self.models.run_swapper(disc, latent, swapper_out)
                    swapper_out = torch.squeeze(swapper_out).permute(1, 2, 0)
                    output[j::dim, i::dim] = swapper_out.clone()
            prev_face = input_face_affined.clone()
            input_face_affined = output.clone()
            output = torch.mul(output, 255)
            output = torch.clamp(output, 0, 255)

        output = output.permute(2, 0, 1)
        swap = t512(output)

        if parameters['StrengthSwitch']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthSlider'], 100) * 0.01
                if alpha == 0:
                    alpha = 1
                prev_face = torch.mul(prev_face, 255).clamp(0, 255).permute(2, 0, 1)
                prev_face = t512(prev_face)
                swap = torch.add(torch.mul(swap, alpha), torch.mul(prev_face, 1 - alpha))

        if parameters['ColorSwitch']:
            swap = torch.unsqueeze(swap, 0)
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaSlider'], 1.0)
            swap = torch.squeeze(swap).permute(1, 2, 0).type(torch.float32)
            del_color = torch.tensor(
                [parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']],
                device=device)
            swap += del_color
            swap = torch.clamp(swap, 0., 255.).permute(2, 0, 1).type(torch.uint8)

        border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        border_mask = torch.unsqueeze(border_mask, 0)
        top = parameters['BorderTopSlider']
        left = parameters['BorderSidesSlider']
        right = 128 - parameters['BorderSidesSlider']
        bottom = 128 - parameters['BorderBottomSlider']
        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0
        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider'] * 2 + 1,
                                        (parameters['BorderBlurSlider'] + 1) * 0.2)
        border_mask = gauss(border_mask)

        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        swap_mask = torch.unsqueeze(swap_mask, 0)

        if parameters['DiffSwitch']:
            mask = self._apply_fake_diff(swap, original_face_512, parameters['DiffSlider'])
            gauss = transforms.GaussianBlur(parameters['BlendSlider'] * 2 + 1,
                                            (parameters['BlendSlider'] + 1) * 0.2)
            mask = gauss(mask.type(torch.float32))
            swap = swap * mask + original_face_512 * (1 - mask)

        if parameters['RestorerSwitch']:
            swap = self._apply_restorer(swap, parameters)

        if parameters['OccluderSwitch']:
            mask = self._apply_occlusion(original_face_256, parameters['OccluderSlider'])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        if parameters['FaceParserSwitch']:
            mask = self._apply_face_parser(swap, parameters['FaceParserSlider'],
                                           parameters['MouthParserSlider'])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        gauss = transforms.GaussianBlur(parameters['BlendSlider'] * 2 + 1,
                                        (parameters['BlendSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        # Composite back into the crop image
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0, 0], [0, 511], [511, 0], [511, 511]])
        x = IM512[0][0] * corners[:, 0] + IM512[0][1] * corners[:, 1] + IM512[0][2]
        y = IM512[1][0] * corners[:, 0] + IM512[1][1] * corners[:, 1] + IM512[1][2]

        c_left = max(floor(np.min(x)), 0)
        c_top = max(floor(np.min(y)), 0)
        c_right = min(ceil(np.max(x)), img.shape[2])
        c_bottom = min(ceil(np.max(y)), img.shape[1])

        pad_w = max(img.shape[2] - 512, 0)
        pad_h = max(img.shape[1] - 512, 0)

        swap = v2.functional.pad(swap, (0, 0, pad_w, pad_h))
        swap = v2.functional.affine(
            swap, tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale, 0,
            interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
        swap = swap[0:3, c_top:c_bottom, c_left:c_right].permute(1, 2, 0)

        swap_mask = v2.functional.pad(swap_mask, (0, 0, pad_w, pad_h))
        swap_mask = v2.functional.affine(
            swap_mask, tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale, 0,
            interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
        swap_mask = swap_mask[0:1, c_top:c_bottom, c_left:c_right].permute(1, 2, 0)
        swap_mask = torch.sub(1, swap_mask)

        img_crop = img[0:3, c_top:c_bottom, c_left:c_right].permute(1, 2, 0)
        img_crop = torch.mul(swap_mask, img_crop)
        swap = torch.add(swap, img_crop).type(torch.uint8).permute(2, 0, 1)
        img[0:3, c_top:c_bottom, c_left:c_right] = swap

        return img

    def _apply_fake_diff(self, swapped_face, original_face, amount):
        diff = swapped_face.permute(1, 2, 0) - original_face.permute(1, 2, 0)
        diff = torch.abs(diff)
        fthresh = amount * 2.55
        diff[diff < fthresh] = 0
        diff[diff >= fthresh] = 1
        diff = torch.sum(diff, dim=2, keepdim=True)
        diff[diff > 0] = 1
        return diff.permute(2, 0, 1)

    def _apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)
        outpred = torch.ones((256, 256), dtype=torch.float32, device=device).contiguous()
        self.models.run_occluder(img, outpred)
        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0).unsqueeze(0).type(torch.float32)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        if amount > 0:
            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
        elif amount < 0:
            outpred = 1 - outpred
            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)
            outpred = torch.squeeze(outpred)
            outpred = 1 - outpred
        else:
            outpred = torch.squeeze(outpred)
        return outpred.reshape(1, 256, 256)

    def _apply_face_parser(self, img, face_amount, mouth_amount):
        img_n = torch.div(img, 255)
        img_n = v2.functional.normalize(img_n, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img_n = img_n.reshape(1, 3, 512, 512)
        outpred = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=device).contiguous()
        self.models.run_faceparser(img_n, outpred)
        outpred = torch.argmax(torch.squeeze(outpred), 0)

        mouth_parse = torch.ones((1, 512, 512), dtype=torch.float32, device=device)
        if mouth_amount != 0:
            idxs = torch.tensor([11, 12, 13] if mouth_amount > 0 else [11], device=device)
            mp = torch.isin(outpred, idxs)
            mp = (~mp).float().reshape(1, 1, 512, 512)
            mp = 1 - mp
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
            for _ in range(int(abs(mouth_amount))):
                mp = torch.nn.functional.conv2d(mp, kernel, padding=(1, 1))
                mp = torch.clamp(mp, 0, 1)
            mp = 1 - torch.squeeze(mp)
            mouth_parse = mp.reshape(1, 512, 512)

        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=device)
        bg = torch.isin(outpred, bg_idxs)
        bg = (~bg).float().reshape(1, 1, 512, 512)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        if face_amount > 0:
            for _ in range(int(face_amount)):
                bg = torch.nn.functional.conv2d(bg, kernel, padding=(1, 1))
                bg = torch.clamp(bg, 0, 1)
            bg = torch.squeeze(bg)
        elif face_amount < 0:
            bg = 1 - bg
            for _ in range(int(-face_amount)):
                bg = torch.nn.functional.conv2d(bg, kernel, padding=(1, 1))
                bg = torch.clamp(bg, 0, 1)
            bg = torch.squeeze(bg)
            bg = 1 - bg
            bg = bg.reshape(1, 512, 512)
        else:
            bg = torch.ones((1, 512, 512), dtype=torch.float32, device=device)

        return torch.mul(bg, mouth_parse)

    def _apply_restorer(self, swapped_face, parameters):
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)

        temp = swapped_face
        if parameters['RestorerDetTypeTextSel'] == 'Blend':
            dst = self.arcface_dst * 4.0
            dst[:, 0] += 32.0
            tform = trans.SimilarityTransform()
            tform.estimate(dst, self.FFHQ_kps)
            temp = v2.functional.affine(
                swapped_face, tform.rotation * 57.2958,
                (tform.translation[0], tform.translation[1]),
                tform.scale, 0, center=(0, 0))
            temp = v2.functional.crop(temp, 0, 0, 512, 512)
        elif parameters['RestorerDetTypeTextSel'] == 'Reference':
            try:
                dst = self.models.resnet50(swapped_face, score=0.5)
                tform = trans.SimilarityTransform()
                tform.estimate(dst, self.FFHQ_kps)
                temp = v2.functional.affine(
                    swapped_face, tform.rotation * 57.2958,
                    (tform.translation[0], tform.translation[1]),
                    tform.scale, 0, center=(0, 0))
                temp = v2.functional.crop(temp, 0, 0, 512, 512)
            except Exception:
                return swapped_face

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            temp = t256(temp)
        temp = torch.unsqueeze(temp, 0).contiguous()

        outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
        rtype = parameters['RestorerTypeTextSel']
        if rtype == 'GFPGAN':
            self.models.run_GFPGAN(temp, outpred)
        elif rtype == 'CF':
            self.models.run_codeformer(temp, outpred)
        elif rtype == 'GPEN256':
            outpred = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_256(temp, outpred)
        elif rtype == 'GPEN512':
            self.models.run_GPEN_512(temp, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = (outpred + 1) / 2 * 255
        if rtype == 'GPEN256':
            outpred = t512(outpred)

        if parameters['RestorerDetTypeTextSel'] in ('Blend', 'Reference'):
            outpred = v2.functional.affine(
                outpred, tform.inverse.rotation * 57.2958,
                (tform.inverse.translation[0], tform.inverse.translation[1]),
                tform.inverse.scale, 0,
                interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))

        alpha = float(parameters['RestorerSlider']) / 100.0
        return torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face, 1 - alpha))
