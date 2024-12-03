import sys
import time

import cv2 as cv
import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module


class MinSolver:

    def min_sample_size(self):
        raise NotImplementedError()

    def solve(self, X, x):
        raise NotImplementedError()


class CVP3P_solver(MinSolver):

    Id = np.eye(3, dtype=np.float32)

    # cv.SOLVEPNP_P3P, cv.SOLVEPNP_AP3P
    def __init__(self, cv_flag=cv.SOLVEPNP_P3P):
        self.cv_flag = cv_flag

    # TODO batched naive impl!, also in numpy on CPU
    def solve(self, X, x):
        """
        :param X: :math:`(B, 3, 3)`
        :param x: :math:`(B, 3, 3)` - homogeneous coordinates
        :return: :math:`(B2, 3, 3)` - new batch size B2!!!
        """
        device: torch.device = X.device
        X_np: np.ndarray = X.detach().cpu().float().numpy().astype(np.float32)
        x_np: np.ndarray = x.detach().cpu().float().numpy().astype(np.float32)

        r = []
        for b in range(X.shape[0]):
            X_in_cv = np.ascontiguousarray(X_np[b, :, :3].reshape((3, 1, 3)))
            x_in_cv = np.ascontiguousarray(x_np[b, :, :2].reshape((3, 1, 2)))
            count, rot_vecs, ts = cv.solveP3P(X_in_cv, x_in_cv, CVP3P_solver.Id, None, self.cv_flag)
            Rs = [cv.Rodrigues(rv)[0] for rv in rot_vecs]
            r.extend([np.hstack((r, t)) for r, t in zip(Rs, ts)])
        return torch.tensor(np.array(r)).to(device=device)

    def min_sample_size(self):
        return 3

class P3P_RANSAC(Module):
    """Module for robust geometry estimation with RANSAC. https://en.wikipedia.org/wiki/Random_sample_consensus.

    Args:
        model_type: type of model to estimate: "homography", "fundamental", "fundamental_7pt",
            "homography_from_linesegments".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """

    def __init__(
        self,
        minimal_solver: MinSolver,
        batch_size: int = 2048,
        max_iter: int = int(1e5),
        confidence: float = 0.99,
        max_lo_iters: int = 5,
        inl_th: float = 2.0,
    ) -> None:
        super().__init__()
        self.minimal_solver = minimal_solver
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.inl_th_sq = inl_th ** 2

        self.minimal_sample_size = minimal_solver.min_sample_size()
        self.elapsed_time_in_solver = 0.0


    def sample(self, sample_size: int, pop_size: int, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp.
        on GPU.
        """
        if device is None:
            device = torch.device("cpu")
        rand = torch.rand(batch_size, pop_size, device=device)
        _, out = rand.topk(k=sample_size, dim=1)
        return out

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        eps = 1e-9
        if num_tc <= sample_size:
            return 1.0
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / min(-eps, math.log(max(eps, 1.0 - math.pow(n_inl / num_tc, sample_size))))

    # TODO to the solver?
    def score_models(self, X: torch.Tensor, x: torch.Tensor, models: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        :param X: :math:`(N, 4)` - homogeneous coordinates !!!
        :param x: :math:`(N, 3)` - homogeneous coordinates !!!
        :param models: :math:`(B, 3, 4)` ([B, (R, t)]).
        :return:
        """
        # if len(x.shape) == self.minimal_solver.min_sample_size():
        #     x = x[None]
        # x = x[None]

        batch_size = models.shape[0]
        X = X.expand(batch_size, -1, -1).permute(0, 2, 1) # (B, 4, N)
        x = x.expand(batch_size, -1, -1).permute(0, 2, 1) # (B, 3, N)

        projected = (models @ X).permute(0, 2, 1) # (B, 3, 4) @ (B, 4, N) => (B, 3, N) => (B, N, 3)
        mask_chirality = projected[..., 2] > 0 # (B, N)
        #print(f"mask_chirality: {(~mask_chirality).sum()}")
        projected[projected[..., 2] == 0][:, 2] = -1.0
        projected = projected / projected[..., 2:3]

        diff_proj = projected - x.permute(0, 2, 1) # (B, N, 3)
        score_vector = torch.linalg.vector_norm(diff_proj, dim=2) ** 2 # (B, N)
        #print(f"max score in vector: {score_vector.topk(k=10, dim=1)[0]}")
        mask_err = score_vector < self.inl_th_sq
        inliers = torch.logical_and(mask_chirality, mask_err) # (B, N)

        score_vector = torch.clip(score_vector, min=0, max=self.inl_th_sq) # (B, N)
        score_vector[torch.logical_not(mask_chirality)] = self.inl_th_sq # (B, N)
        models_score = score_vector.sum(dim=1) # (B)

        best_model_idx = models_score.argmin()
        best_model_score = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inliers[best_model_idx].clone()
        return model_best, inliers_best, best_model_score

    @staticmethod
    def to_homogeneous(a: torch.Tensor, to_dim: int) -> torch.Tensor:
        if a.shape[-1] == to_dim:
            return a
        elif a.shape[-1] == to_dim - 1:
            return torch.nn.functional.pad(a, [0, 1], "constant", 1.0)
        else:
            raise ValueError(f"{a.shape[-1]} vs. {to_dim}")

    def validate_inputs(self, X: torch.Tensor, x: torch.Tensor) -> None:

        dim1 = X.shape[-1]
        assert dim1 in [3, 4]
        if dim1 == 4:
            assert torch.all(1.0 == X[:, 3])

        dim2 = x.shape[-1]
        assert dim2 in [2, 3]
        if dim2 == 3:
            assert torch.all(1.0 == x[:, 2])

        assert X.shape[0] == x.shape[0]
        assert X.shape[0] >= self.minimal_sample_size

    def forward(self, X: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm. Device inferred now from the data.

        Args:
            x: 2D corrs :math:`(N, 2)`. Assumed to be already normalized by K.inv(): torch.Tensor
            X: 3D corrs :math:`(N, 3)`.

        Returns:
            - Estimated model, shape of (:math:`(3, 3)`, :math:`(3)`).
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
        """

        self.validate_inputs(X, x)
        #print(f"solving for {len(X)} points")
        X = P3P_RANSAC.to_homogeneous(X, 4)
        x = P3P_RANSAC.to_homogeneous(x, 3)
        best_score: float = sys.float_info.max
        num_tc: int = len(x)
        best_model_Rt = torch.zeros(3, 4, dtype=x.dtype, device=x.device)
        best_inliers: torch.Tensor = torch.zeros(num_tc, 1, device=x.device, dtype=torch.bool)

        for i in range(self.max_iter):

            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size, x.device)
            pts_2d_sampled = x[idxs]
            pts_3d_sampled = X[idxs]

            start = time.time()
            models = self.minimal_solver.solve(pts_3d_sampled, pts_2d_sampled)
            end = time.time()
            self.elapsed_time_in_solver += end - start
            if len(models) == 0:
                continue

            # Score the models and select the best one
            model, inliers, model_score = self.score_models(X, x, models)

            # Store far-the-best model
            if model_score < best_score:

                in_ratio = inliers.sum() / inliers.shape[0]
                #print(f"inlier ratio when updating: {in_ratio * 100}%")

                # TODO no local optimization
                # Store the best model
                best_model_Rt = model
                best_inliers = inliers
                best_score = model_score

                # Should we already stop?
                # TODO confirm the best score is actually the number of the inlers,
                # and not the continuous number (which should be considered when picking the best model?)
                new_max_iter = int(
                    self.max_samples_by_conf(best_inliers.sum().item(), num_tc, self.minimal_sample_size, self.confidence)
                )
                # Stop estimation, if the model is very good
                if (i + 1) * self.batch_size >= new_max_iter:
                    break

        #print(f"{i} iterations for {self.batch_size=}")
        return best_model_Rt, best_inliers


if __name__ == "__main__":

    min_solver = CVP3P_solver(cv_flag=cv.SOLVEPNP_P3P)
    ransac = P3P_RANSAC(minimal_solver=min_solver, batch_size=1)
    # TODO test on something?
    # ransac.forward(None, None)
