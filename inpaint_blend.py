import torch
import torch.nn.functional as F
import numpy as np

MAX_RESOLUTION = 16384

class InpaintBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "blend_mode": (["default", "poisson"], {"default": "default"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "image/inpainting"

    def composite(self, destination, source, x, y, resize_source, blend_mode="default", mask=None):
        if blend_mode == "default":
            return self.default_composite(destination, source, x, y, resize_source, mask)
        elif blend_mode == "poisson":
            return self.poisson_composite(destination, source, x, y, resize_source, mask)

    # -------------------------------------------------------------
    # DEFAULT: straight alpha/mask compositing (pure PyTorch)
    # -------------------------------------------------------------
    def default_composite(self, destination, source, x, y, resize_source, mask=None):
        assert destination.ndim == 4 and source.ndim == 4
        Bd, Hd, Wd, Cd = destination.shape
        Bs, Hs, Ws, Cs = source.shape
        assert Cd in (3, 4) and Cs in (3, 4)

        B = max(Bd, Bs)
        if Bd not in (1, B) or Bs not in (1, B):
            raise ValueError(f"Batch sizes not compatible: destination={Bd}, source={Bs}")

        if mask is not None:
            assert mask.ndim == 3
            Bm, Hm, Wm = mask.shape
            if (Hm, Wm) != (Hs, Ws):
                if not resize_source:
                    raise ValueError(f"mask {Hm}x{Wm} must match source {Hs}x{Ws} (or enable resize_source).")
                mask = self._resize_mask_t(mask, (Ws, Hs))
            if Bm not in (1, B):
                raise ValueError(f"Batch sizes not compatible: mask={Bm}, expected 1 or {B}")

        out_frames = []
        for i in range(B):
            d = destination[i if Bd > 1 else 0][..., :3]
            s = source[i if Bs > 1 else 0][..., :3]
            a = source[i if Bs > 1 else 0][..., 3] if source.shape[-1] == 4 else None
            m = mask[i if (mask is not None and mask.shape[0] > 1) else 0] if mask is not None else None

            if resize_source and (x < 0 or y < 0 or x + s.shape[1] > d.shape[1] or y + s.shape[0] > d.shape[0]):
                # scale source/mask to fit starting at (x,y)
                max_w = max(1, min(d.shape[1] - max(0, x), s.shape[1]))
                max_h = max(1, min(d.shape[0] - max(0, y), s.shape[0]))
                scale = min(max_w / s.shape[1], max_h / s.shape[0])
                new_w = max(1, int(round(s.shape[1] * scale)))
                new_h = max(1, int(round(s.shape[0] * scale)))
                s = self._resize_img_torch(s, new_w, new_h)
                if a is not None: a = self._resize_mask_t(a, (new_w, new_h))
                if m is not None: m = self._resize_mask_t(m, (new_w, new_h))

            # crop to fit
            s, a, m, dx0, dy0, dx1, dy1 = self._crop_pack(d, s, a, m, x, y)
            if s.numel() == 0:
                out_frames.append(d)
                continue

            if a is None and m is None:
                eff = torch.ones((s.shape[0], s.shape[1]), dtype=d.dtype, device=d.device)
            elif a is None:
                eff = torch.clamp(m, 0, 1)
            elif m is None:
                eff = torch.clamp(a, 0, 1)
            else:
                eff = torch.clamp(a, 0, 1) * torch.clamp(m, 0, 1)

            roi = d[dy0:dy1, dx0:dx1, :]
            blended = eff.unsqueeze(-1) * s + (1.0 - eff.unsqueeze(-1)) * roi
            d_out = d.clone()
            d_out[dy0:dy1, dx0:dx1, :] = blended
            out_frames.append(d_out)

        return (torch.stack(out_frames, dim=0),)

    # -------------------------------------------------------------
    # POISSON: gradient-domain blending (Jacobi, pure PyTorch)
    # -------------------------------------------------------------
    def poisson_composite(self, destination, source, x, y, resize_source, mask=None, iters: int = 400, tol: float = 1e-4):
        """
        True Poisson image editing (Pérez et al. 2003).
        - Uses source gradients (normal cloning).
        - Enforces destination boundary; no OpenCV, no feathered ellipse.
        - (x, y) is TOP-LEFT where the source patch lands in destination.
        """
        assert destination.ndim == 4 and source.ndim == 4
        Bd, Hd, Wd, Cd = destination.shape
        Bs, Hs, Ws, Cs = source.shape
        assert Cd in (3, 4) and Cs in (3, 4)
        B = max(Bd, Bs)
        if Bd not in (1, B) or Bs not in (1, B):
            raise ValueError(f"Batch sizes not compatible: destination={Bd}, source={Bs}")

        if mask is None and Cs == 4:
            mask = source[..., 3].clone()
        if mask is None:
            # if nothing provided, use full patch
            mask = torch.ones((B if Bs > 1 else 1, Hs, Ws), dtype=destination.dtype, device=destination.device)

        # Ensure mask size matches source (resize if allowed)
        Bm, Hm, Wm = mask.shape
        if (Hm, Wm) != (Hs, Ws):
            if not resize_source:
                raise ValueError(f"mask {Hm}x{Wm} must match source {Hs}x{Ws} (or enable resize_source).")
            mask = self._resize_mask_t(mask, (Ws, Hs))
        if Bm not in (1, B):
            raise ValueError(f"Batch sizes not compatible: mask={Bm}, expected 1 or {B}")

        out_frames = []
        for i in range(B):
            d = destination[i if Bd > 1 else 0][..., :3].contiguous()
            s = source[i if Bs > 1 else 0][..., :3].contiguous()
            m = mask[i if (mask is not None and mask.shape[0] > 1) else 0].contiguous()

            if resize_source and (x < 0 or y < 0 or x + s.shape[1] > d.shape[1] or y + s.shape[0] > d.shape[0]):
                max_w = max(1, min(d.shape[1] - max(0, x), s.shape[1]))
                max_h = max(1, min(d.shape[0] - max(0, y), s.shape[0]))
                scale = min(max_w / s.shape[1], max_h / s.shape[0])
                new_w = max(1, int(round(s.shape[1] * scale)))
                new_h = max(1, int(round(s.shape[0] * scale)))
                s = self._resize_img_torch(s, new_w, new_h)
                m = self._resize_mask_t(m, (new_w, new_h))

            # crop to fit
            s, _, m, dx0, dy0, dx1, dy1 = self._crop_pack(d, s, None, m, x, y)
            if s.numel() == 0:
                out_frames.append(d)
                continue

            # binarize mask (strict Poisson region)
            m = (m > 0.5).to(d.dtype)

            # destination ROI
            d_roi = d[dy0:dy1, dx0:dx1, :]

            # Solve inside mask
            blended_roi = self._poisson_solve_roi(d_roi, s, m, iters=iters, tol=tol)

            # write back
            d_out = d.clone()
            d_out[dy0:dy1, dx0:dx1, :] = torch.where(m.unsqueeze(-1) > 0, blended_roi, d_roi)
            out_frames.append(d_out)

        return (torch.stack(out_frames, dim=0),)

    # -------------------------------------------------------------
    # Internal: Poisson solver on a single ROI (H,W,3)
    # -------------------------------------------------------------
    def _poisson_solve_roi(self, dst: torch.Tensor, src: torch.Tensor, mask: torch.Tensor, iters=400, tol=1e-4):
        """
        Jacobi iterations for:   ΔV = ΔS  inside Ω;   V = D on ∂Ω
        dst, src: (H,W,3) float in [0,1]
        mask:     (H,W)   1 inside Ω, 0 outside
        """
        device = dst.device
        dtype  = dst.dtype
        H, W = dst.shape[:2]

        # Precompute laplacian of src (guidance) with replicate padding
        lap = self._laplacian(src)  # (H,W,3)

        # Initial guess = destination
        V = dst.clone()

        # convenience: bool mask and neighbors
        M  = (mask > 0.5)
        M_up    = self._shift_mask(M, 0, -1)
        M_down  = self._shift_mask(M, 0,  1)
        M_left  = self._shift_mask(M, -1, 0)
        M_right = self._shift_mask(M,  1, 0)

        for _ in range(iters):
            V_up    = self._shift_img(V, 0, -1)
            V_down  = self._shift_img(V, 0,  1)
            V_left  = self._shift_img(V, -1, 0)
            V_right = self._shift_img(V,  1, 0)

            D_up    = self._shift_img(dst, 0, -1)
            D_down  = self._shift_img(dst, 0,  1)
            D_left  = self._shift_img(dst, -1, 0)
            D_right = self._shift_img(dst,  1, 0)

            # Neighbor sum: use V for neighbors inside Ω, else D
            sumN = torch.where(M_up.unsqueeze(-1),    V_up,    D_up)
            sumN = sumN + torch.where(M_down.unsqueeze(-1),  V_down,  D_down)
            sumN = sumN + torch.where(M_left.unsqueeze(-1),  V_left,  D_left)
            sumN = sumN + torch.where(M_right.unsqueeze(-1), V_right, D_right)

            V_new = (sumN + lap) / 4.0

            # Keep only inside Ω; outside stays destination
            V_new = torch.where(M.unsqueeze(-1), V_new, dst)

            # early stop
            err = torch.mean(torch.abs(V_new - V)[M.unsqueeze(-1).expand_as(V)])
            V = V_new
            if err.item() < tol:
                break

        return torch.clamp(V, 0.0, 1.0)

    # -------------------------------------------------------------
    # helpers (resize/crop/shift/laplacian)
    # -------------------------------------------------------------
    def _resize_img_torch(self, img: torch.Tensor, new_w: int, new_h: int) -> torch.Tensor:
        img_bchw = img.permute(2, 0, 1).unsqueeze(0)
        img_bchw = F.interpolate(img_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return img_bchw.squeeze(0).permute(1, 2, 0)

    def _resize_mask_t(self, mask_t: torch.Tensor, new_size_wh):
        newW, newH = new_size_wh
        m = mask_t.unsqueeze(1)  # (B,1,H,W) or (H,W)->(1,1,H,W)
        if m.ndim == 3: m = m.unsqueeze(0)
        m = F.interpolate(m, size=(newH, newW), mode="nearest")
        m = m.squeeze(1)
        return m

    def _crop_pack(self, d_rgb, s_rgb, a, m, x, y):
        Hd, Wd = d_rgb.shape[:2]
        Hs, Ws = s_rgb.shape[:2]
        x0, y0 = x, y
        x1, y1 = x + Ws, y + Hs

        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(Wd, x1)
        dy1 = min(Hd, y1)
        sx0 = dx0 - x0
        sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0)
        sy1 = sy0 + (dy1 - dy0)
        if dx0 >= dx1 or dy0 >= dy1:
            z = d_rgb.new_zeros((0, 0, 3))
            z2 = d_rgb.new_zeros((0, 0))
            return z, (None if a is None else z2), (None if m is None else z2), dx0, dy0, dx1, dy1

        s_crop = s_rgb[sy0:sy1, sx0:sx1, :]
        a_crop = None if a is None else a[sy0:sy1, sx0:sx1]
        m_crop = None if m is None else m[sy0:sy1, sx0:sx1]
        return s_crop, a_crop, m_crop, dx0, dy0, dx1, dy1

    def _shift_img(self, img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        # replicate padding to keep same size
        img_bchw = img.permute(2,0,1).unsqueeze(0)
        pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))  # (left,right,top,bottom) for F.pad is (left,right,top,bottom)
        img_pad = F.pad(img_bchw, pad, mode='replicate')
        x0 = max(-dx,0)
        x1 = x0 + img.shape[1]
        y0 = max(-dy,0)
        y1 = y0 + img.shape[0]
        out = img_pad[:, :, y0:y1, x0:x1]
        return out.squeeze(0).permute(1,2,0)

    def _shift_mask(self, mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        m = mask.unsqueeze(0).unsqueeze(0).float()
        pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))
        m_pad = F.pad(m, pad, mode='replicate')
        x0 = max(-dx,0)
        x1 = x0 + mask.shape[1]
        y0 = max(-dy,0)
        y1 = y0 + mask.shape[0]
        return (m_pad[:, :, y0:y1, x0:x1].squeeze(0).squeeze(0) > 0.5)

    def _laplacian(self, img: torch.Tensor) -> torch.Tensor:
        # 4-neighbor discrete Laplacian with replicate padding
        up    = self._shift_img(img, 0, -1)
        down  = self._shift_img(img, 0,  1)
        left  = self._shift_img(img, -1, 0)
        right = self._shift_img(img,  1, 0)
        return 4*img - up - down - left - right
