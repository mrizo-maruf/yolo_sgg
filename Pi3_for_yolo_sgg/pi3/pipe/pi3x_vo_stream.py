from ..utils.geometry import depth_edge
import torch
import torch.nn.functional as F


class Pi3XVOStream:
    """
    Streaming version of Pi3XVO.

    Instead of receiving all frames at once, frames are pushed one at a time
    via `push_frame()`. When enough frames accumulate (reaching `chunk_size`),
    inference runs automatically and results are yielded.

    Usage:
        stream = Pi3XVOStream(model, chunk_size=30, overlap=10)

        for frame in camera_source:
            results = stream.push_frame(frame)
            if results is not None:
                # results['depth']:  (new_frames, H, W) depth maps
                # results['poses']:  (new_frames, 4, 4) camera poses
                # results['conf']:   (new_frames, H, W) confidence
                # results['points']: (new_frames, H, W, 3) global 3D points
                process(results)

        # Flush remaining buffered frames
        results = stream.flush()
        if results is not None:
            process(results)
    """

    def __init__(
        self,
        model,
        chunk_size=30,
        overlap=10,
        conf_thre=0.05,
        inject_condition=None,
        intrinsics=None,
        dtype=torch.bfloat16,
    ):
        """
        Args:
            model: Pi3X model (already on device, eval mode).
            chunk_size: Number of frames per inference chunk.
            overlap: Number of overlapping frames between chunks for Sim3 alignment.
            conf_thre: Confidence threshold for masking low-quality points.
            inject_condition: List of conditions to inject from previous chunk
                              (e.g. ['pose', 'depth']). Default: no injection.
            intrinsics: Camera intrinsics tensor, shape (1, 1, 3, 3).
                        If provided, will be broadcast to all frames.
                        Can also be set per-frame via push_frame().
            dtype: Inference dtype (bfloat16 or float16).
        """
        self.model = model
        self.model.eval()

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.conf_thre = conf_thre
        self.inject_condition = inject_condition or []
        self.dtype = dtype

        # Shared intrinsics (1, 1, 3, 3) - broadcast to all frames
        self.intrinsics = intrinsics

        # Frame buffer: list of (3, H, W) tensors on device
        self._buffer = []
        # Per-frame intrinsics buffer (optional)
        self._intrinsics_buffer = []

        # Alignment state from previous chunk
        self._prev_global_pts_overlap = None
        self._prev_global_mask_overlap = None
        self._prev_aligned_poses_overlap = None
        self._prev_local_depth_overlap = None
        self._prev_local_conf_overlap = None
        self._prev_rays_overlap = None

        self._is_first_chunk = True
        self._total_frames_emitted = 0
        self._flushed = False

    def push_frame(self, frame, intrinsics=None):
        """
        Push a single RGB frame into the buffer.

        Args:
            frame: RGB image tensor, shape (3, H, W), values in [0, 1], on device.
            intrinsics: Optional per-frame intrinsics (3, 3) tensor.
                        Overrides the shared intrinsics for this frame.

        Returns:
            dict or None: If the buffer reached chunk_size, runs inference and
            returns results for the NEW (non-overlap) frames. Otherwise None.
        """
        if self._flushed:
            raise RuntimeError("Stream already flushed. Create a new instance.")

        self._buffer.append(frame)
        self._intrinsics_buffer.append(intrinsics)

        if len(self._buffer) >= self.chunk_size:
            return self._process_chunk()

        return None

    def flush(self):
        """
        Process any remaining frames in the buffer.

        Returns:
            dict or None: Results for remaining frames, or None if buffer is empty.
        """
        if self._flushed:
            return None
        self._flushed = True

        if len(self._buffer) == 0:
            return None

        # If this is the first chunk and buffer is small, just run it
        # If not first chunk, need at least overlap+1 frames to produce new output
        if not self._is_first_chunk and len(self._buffer) <= self.overlap:
            return None

        return self._process_chunk()

    def _build_intrinsics_tensor(self, n_frames, device):
        """Build intrinsics tensor (1, n_frames, 3, 3) from available sources."""
        # Check per-frame intrinsics first
        per_frame = self._intrinsics_buffer[:n_frames]
        has_per_frame = any(k is not None for k in per_frame)

        if has_per_frame:
            # Use per-frame intrinsics, fall back to shared for missing frames
            Ks = []
            for k in per_frame:
                if k is not None:
                    Ks.append(k)
                elif self.intrinsics is not None:
                    Ks.append(self.intrinsics[0, 0])
                else:
                    return None  # Can't build complete intrinsics
            return torch.stack(Ks).unsqueeze(0)  # (1, N, 3, 3)

        if self.intrinsics is not None:
            return self.intrinsics.expand(1, n_frames, -1, -1)

        return None

    @torch.no_grad()
    def _process_chunk(self):
        """Run inference on the current buffer and return new-frame results."""
        device = self._buffer[0].device
        chunk_imgs = torch.stack(self._buffer).unsqueeze(0)  # (1, N, 3, H, W)
        current_len = chunk_imgs.shape[1]
        B, N, C, H, W = chunk_imgs.shape

        print(f"[Pi3XVOStream] Processing chunk: {current_len} frames "
              f"(total emitted so far: {self._total_frames_emitted})")

        # Build model kwargs
        model_kwargs = {'with_prior': False}

        intrinsics_seq = self._build_intrinsics_tensor(current_len, device)
        if intrinsics_seq is not None:
            model_kwargs['intrinsics'] = intrinsics_seq
            model_kwargs['with_prior'] = True

        # Inject conditions from previous chunk overlap
        if not self._is_first_chunk:
            overlap = self.overlap

            if 'pose' in self.inject_condition and self._prev_aligned_poses_overlap is not None:
                prior_poses = torch.eye(4, device=device).repeat(1, current_len, 1, 1)
                prior_poses[:, :overlap] = self._prev_aligned_poses_overlap
                mask_pose = torch.zeros((1, current_len), dtype=torch.bool, device=device)
                mask_pose[:, :overlap] = True
                model_kwargs['poses'] = prior_poses
                model_kwargs['mask_add_pose'] = mask_pose
                model_kwargs['with_prior'] = True

            if 'depth' in self.inject_condition and self._prev_local_depth_overlap is not None:
                prior_depths = torch.zeros((1, current_len, H, W), device=device)
                prior_depths[:, :overlap] = self._prev_local_depth_overlap
                mask_depth = torch.zeros((1, current_len), dtype=torch.bool, device=device)
                mask_depth[:, :overlap] = True
                if self._prev_local_conf_overlap is not None:
                    valid_mask = self._prev_local_conf_overlap > self.conf_thre
                    prior_depths[:, :overlap][~valid_mask] = 0
                model_kwargs['depths'] = prior_depths
                model_kwargs['mask_add_depth'] = mask_depth
                model_kwargs['with_prior'] = True

            if ('ray' in self.inject_condition or 'intrinsic' in self.inject_condition) \
                    and self._prev_rays_overlap is not None:
                prior_rays = torch.zeros((1, current_len, H, W, 3), device=device)
                prior_rays[:, :overlap] = self._prev_rays_overlap
                mask_ray = torch.zeros((1, current_len), dtype=torch.bool, device=device)
                mask_ray[:, :overlap] = True
                model_kwargs['rays'] = prior_rays
                model_kwargs['mask_add_ray'] = mask_ray
                model_kwargs['with_prior'] = True

        # --- Inference ---
        with torch.amp.autocast('cuda', dtype=self.dtype):
            pred = self.model(chunk_imgs, **model_kwargs)

        curr_local_depth = pred['local_points'][..., 2]
        curr_pts = pred['points']
        curr_poses = pred['camera_poses']
        curr_conf = torch.sigmoid(pred['conf'])[..., 0]
        curr_rays = pred['rays']

        edge = depth_edge(curr_local_depth, rtol=0.03)
        curr_conf[edge] = 0
        curr_mask = curr_conf > self.conf_thre

        if curr_mask.sum() < 10:
            flat_conf = curr_conf.view(1, current_len, -1)
            k = int(flat_conf.shape[-1] * 0.1)
            topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
            min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
            curr_mask = curr_conf >= min_vals

        # --- Sim3 alignment ---
        if self._is_first_chunk:
            aligned_pts = curr_pts
            aligned_poses = curr_poses
        else:
            overlap = self.overlap
            src_pts = curr_pts[:, :overlap]
            src_mask = curr_mask[:, :overlap]
            tgt_pts = self._prev_global_pts_overlap
            tgt_mask = self._prev_global_mask_overlap

            transform_matrix = self._compute_sim3_umeyama_masked(
                src_pts, tgt_pts, src_mask, tgt_mask
            )
            aligned_pts = self._apply_sim3_to_points(curr_pts, transform_matrix)
            aligned_poses = self._apply_sim3_to_poses(curr_poses, transform_matrix)

        # --- Determine which frames are new ---
        if self._is_first_chunk:
            new_start = 0
        else:
            new_start = self.overlap

        new_pts = aligned_pts[:, new_start:]
        new_poses = aligned_poses[:, new_start:]
        new_conf = curr_conf[:, new_start:]
        new_local_depth = curr_local_depth[:, new_start:]

        n_new = new_pts.shape[1]

        # --- Save overlap state for next chunk ---
        self._prev_global_pts_overlap = aligned_pts[:, -self.overlap:]
        self._prev_global_mask_overlap = curr_mask[:, -self.overlap:]
        self._prev_aligned_poses_overlap = aligned_poses[:, -self.overlap:]
        self._prev_local_depth_overlap = curr_local_depth[:, -self.overlap:]
        self._prev_local_conf_overlap = curr_conf[:, -self.overlap:]
        self._prev_rays_overlap = curr_rays[:, -self.overlap:]

        self._is_first_chunk = False

        # --- Slide the buffer: keep only the last `overlap` frames ---
        self._buffer = self._buffer[-self.overlap:]
        self._intrinsics_buffer = self._intrinsics_buffer[-self.overlap:]

        # --- Extract depth from global points ---
        # Transform global points back to local camera space for depth
        points_homo = torch.cat(
            [new_pts[0], torch.ones_like(new_pts[0][..., :1])], dim=-1
        )  # (n_new, H, W, 4)
        poses_inv = torch.inverse(new_poses[0])  # (n_new, 4, 4)
        local_pts = torch.einsum('nij,nhwj->nhwi', poses_inv, points_homo)[..., :3]
        depth_maps = local_pts[..., 2]  # (n_new, H, W)

        self._total_frames_emitted += n_new

        # Cleanup
        del pred, curr_pts, curr_poses, curr_mask, curr_local_depth, curr_conf, curr_rays
        torch.cuda.empty_cache()

        return {
            'points': new_pts[0],        # (n_new, H, W, 3)
            'depth': depth_maps,          # (n_new, H, W)
            'poses': new_poses[0],        # (n_new, 4, 4)
            'conf': new_conf[0],          # (n_new, H, W)
        }

    # --- Sim3 alignment methods (identical to Pi3XVO) ---

    def _compute_sim3_umeyama_masked(self, src_points, tgt_points, src_mask, tgt_mask):
        B = src_points.shape[0]
        device = src_points.device

        src = src_points.reshape(B, -1, 3)
        tgt = tgt_points.reshape(B, -1, 3)

        mask = (src_mask.reshape(B, -1) & tgt_mask.reshape(B, -1)).float().unsqueeze(-1)
        valid_cnt = mask.sum(dim=1).squeeze(-1)
        eps = 1e-6

        bad_mask = valid_cnt < 10
        if bad_mask.all():
            return torch.eye(4, device=device).repeat(B, 1, 1)

        src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)

        src_centered = (src - src_mean) * mask
        tgt_centered = (tgt - tgt_mean) * mask

        H = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
        U, S, V = torch.svd(H)

        R = torch.bmm(V, U.transpose(1, 2))

        det = torch.det(R)
        diag = torch.ones(B, 3, device=device)
        diag[:, 2] = torch.sign(det)
        R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))

        src_var = (src_centered ** 2).sum(dim=2) * mask.squeeze(-1)
        src_var = src_var.sum(dim=1) / (valid_cnt + eps)

        corrected_S = S.clone()
        corrected_S[:, 2] *= diag[:, 2]
        trace_S = corrected_S.sum(dim=1)

        scale = trace_S / (src_var * valid_cnt + eps)
        scale = scale.view(B, 1, 1)

        t = tgt_mean.transpose(1, 2) - scale * torch.bmm(R, src_mean.transpose(1, 2))

        sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
        sim3[:, :3, :3] = scale * R
        sim3[:, :3, 3] = t.squeeze(2)

        if bad_mask.any():
            identity = torch.eye(4, device=device).repeat(B, 1, 1)
            sim3[bad_mask] = identity[bad_mask]

        return sim3

    def _apply_sim3_to_points(self, points, sim3):
        B, T, H, W, C = points.shape
        flat_pts = points.reshape(B, -1, 3)
        R_s = sim3[:, :3, :3]
        t = sim3[:, :3, 3].unsqueeze(1)
        out_pts = torch.bmm(flat_pts, R_s.transpose(1, 2)) + t
        return out_pts.reshape(B, T, H, W, 3)

    def _apply_sim3_to_poses(self, poses, sim3):
        sim3_expanded = sim3.unsqueeze(1)
        return torch.matmul(sim3_expanded, poses)
