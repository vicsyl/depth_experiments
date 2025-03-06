import math
import sys
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import torch
from torch.nn import Module


class BaseP3PSolver:

    Id = np.eye(3, dtype=np.float32)

    # cv.SOLVEPNP_P3P, cv.SOLVEPNP_AP3P
    def __init__(self, cv_flag=cv.SOLVEPNP_P3P):
        self.cv_flag = cv_flag

    def min_sample_size(self):
        return 3

    # TODO batched naive impl!, also in numpy on CPU
    def solve(self, X, x):
        """
        :param X: :math:`(B, 3, 4)` - homogeneous coordinates
        :param x: :math:`(B, 3, 3)` - homogeneous coordinates
        :return: :math:`(B2, 3, 3)` - new batch size B2!!!
        """
        device: torch.device = X.device
        X_np: np.ndarray = X.detach().cpu().float().numpy().astype(np.float32)
        x_np: np.ndarray = x.detach().cpu().float().numpy().astype(np.float32)

        r = []
        samples_X = torch.zeros((0, 3, 4), device=device)
        samples_x = torch.zeros((0, 3, 3), device=device)
        for b in range(X.shape[0]):
            X_in_cv = np.ascontiguousarray(X_np[b, :, :3].reshape((3, 1, 3)))
            x_in_cv = np.ascontiguousarray(x_np[b, :, :2].reshape((3, 1, 2)))
            count, rot_vecs, ts = cv.solveP3P(X_in_cv, x_in_cv, BaseP3PSolver.Id, None, self.cv_flag)
            Rs = [cv.Rodrigues(rv)[0] for rv in rot_vecs]
            to_add = [np.hstack((r, t)) for r, t in zip(Rs, ts)]

            inter = [(i @ X_np[b].T).T for i in to_add]
            mask = [np.all(i[:, 2] > 0) for i in inter]
            # assert np.all(np.array(mask))
            to_add = [i for (i, v) in zip(to_add, mask) if v]

            r.extend(to_add)
            samples_X = torch.cat((samples_X, X[b].expand(len(to_add), -1, -1)), dim=0)
            samples_x = torch.cat((samples_x, x[b].expand(len(to_add), -1, -1)), dim=0)

        r = torch.tensor(np.array(r)).to(device=device)
        return r, samples_X, samples_x


# class ScoreMixin:
#     def score_models(self, X: torch.Tensor, x: torch.Tensor, models: torch.Tensor,
#                      X_sample: torch.Tensor, x_sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
#         raise NotImplementedError()


class P3P_depth_solver(BaseP3PSolver):

    ps = [[0, 1], [1, 2], [2, 0]]

    def __init__(self, depth_map, K, projection_th_sq, depth_th_sq, cv_flag=cv.SOLVEPNP_P3P):
        super(P3P_depth_solver, self).__init__(cv_flag=cv_flag)
        self.depth_map = depth_map
        self.K = K
        self.projection_th_sq = projection_th_sq
        self.depth_th_sq = depth_th_sq

    def score_models(self, X: torch.Tensor, x: torch.Tensor, models: torch.Tensor,
                     X_sample: torch.Tensor, x_sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:

        K = torch.from_numpy(self.K).to(X.device)
        solutions = models.shape[0]
        x_projected = models @ X_sample.permute(0, 2, 1) # .permute(0, 2, 1) # [S, 3, 3]
        # TODO ok I can assert that it is equivalent to x
        solution_depths = x_projected[:, 2, :] # [S, 3] # this way imho without permute

        # list[3] of [S, 1, 2, 1]
        sold_vecs = [torch.cat((solution_depths[:, P3P_depth_solver.ps[i][0:1]], solution_depths[:, P3P_depth_solver.ps[i][1:2]]), dim=1)[:, None, :, None] for i in range(3)]
        # [S, 3, 2, 1]

        sold_vecs = torch.cat(sold_vecs, dim=1)

        camera_coords = (torch.linalg.inv(K).expand(solutions, -1, -1) @ x_sample.permute(0, 2, 1)).permute(0, 2, 1) # [S, 3, 3] = [..., (x,y,1)]
        # chirality check
        assert torch.allclose(camera_coords[..., 2], torch.tensor(1.0).double())
        camera_coords = camera_coords.int()
        depths = self.depth_map[camera_coords[..., 1], camera_coords[..., 0]] # [S, 3]

        depths_lines = [torch.cat((depths[:, i, None], torch.ones_like(depths[:, i])[:, None]), dim=1) for i in range(3)] # list[3] of [S, 2]
        # list[3] of [S, 1, 2, 2]
        depths_mats = [torch.cat((depths_lines[P3P_depth_solver.ps[i][0]][:, None, None], depths_lines[P3P_depth_solver.ps[i][1]][:, None, None]), dim=2) for i in range(3)]
        # [S, 3, 2, 2]
        depths_mats = torch.cat(depths_mats, dim=1)

        depths_mats_inv, info = torch.linalg.inv_ex(depths_mats, check_errors=False)
        # info == 1 <=> singular
        # this is more elegant yet not stable IMHO
        # depths_mats_inv[torch.isnan(depths_mats_inv)] = 1.0
        # this is ugly yet stable IMHO
        # depths_mats_inv[info[..., None, None].expand(-1, -1, 2, 2).bool()] = 1.0
        # TODO how about leaving nans there....

        # [S, 3, 2](solution, permutation, (alpha, beta))
        alpha_betas = depths_mats_inv.double() @ sold_vecs
        alpha_betas = alpha_betas[..., 0]

        # alpha_positive = (abs[..., 0] > 0)
        # mask = ~info.bool() & alpha_positive
        # # mask = mask[..., None].expand(-1, -1, 2).bool()
        # TODO how about leaving nans there
        alpha_betas[alpha_betas[..., 0] > 0] = torch.nan

        # solution_depths[sol, ind] @ (alpha, beta) = depths[0, ind]
        # (alpha, beta) = solution_depths[sol, ind].inv() @ depths[0, ind]

        # now depth of all x => camera_coords_all
        #camera_coords_all = (torch.linalg.inv(K) @ x_sample.permute(0, 2, 1)).permute(0, 2, 1)
        camera_coords_all = (torch.linalg.inv(K) @ x.T).T # [N, 3] = [:, (x,y,1)]
        assert torch.allclose(camera_coords_all[:, 2], torch.tensor(1.0).double())
        cci = camera_coords_all.int()
        depths_all = self.depth_map[cci[:, 1], cci[:, 0]].expand(solutions, 3, -1) # [S, 3, N]
        depths_all = depths_all * alpha_betas[..., 0:1] + alpha_betas[..., 1:2] # [S, 3, N]
        depths_all = depths_all[..., None] # [S, 3, N, 1]
        camera_coords_all = camera_coords_all.expand(solutions, 3, -1, -1) * depths_all # [S, 3(perm), N, 3(cord)]

        # camera_coords_all vs. x_projected_all
        # CONTINUE HERE
        x_projected_all = (models @ X.T.expand(solutions, -1, -1)).permute(0, 2, 1) # [S, N, 3]
        x_projected_all = x_projected_all[:, None].expand(-1, 3, -1, -1) # [S, 3(perm), N, 3(cord)]

        # TODO modulo non-positive values
        camera_coords_ah = x.expand(solutions, -1, -1) # [S, N, 3(cord)]
        x_projected_ah = (K.expand(solutions, -1, -1) @ x_projected_all[:, 0].clone().permute(0, 2, 1)).permute(0, 2, 1) # [S, N, 3(cord)]
        x_projected_ah = x_projected_ah / x_projected_ah[..., 2:3] # [S, N, 3(cord)]

        projection_dist = ((camera_coords_ah - x_projected_ah) ** 2).sum(dim=-1) # [S, N]
        projection_inliers = projection_dist < self.projection_th_sq
        projection_inliers = projection_inliers[:, None].expand(-1, 3, -1)
        projection_dist = torch.clip(projection_dist, min=0, max=self.projection_th_sq)[:, None].expand(-1, 3, -1)

        depth_dist = (camera_coords_all - x_projected_all)[..., 2] ** 2 # [S, 3(perm), N]
        depth_inliers = depth_dist < self.depth_th_sq
        depth_dist = torch.clip(depth_dist, min=0, max=self.depth_th_sq)

        inliers = projection_inliers & depth_inliers # [S, 3(perm), N]

        depth_dist[projection_inliers] = self.depth_th_sq
        projection_dist[depth_inliers] = self.projection_th_sq
        scores = (depth_dist.sum(dim=-1) + projection_dist.sum(dim=-1)) #[S, 3(perm)]

        indexf = scores.argmin()
        index = torch.unravel_index(indexf, scores.shape)
        return models[index[0]], inliers[index], scores[index].item()


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
        batch_size: int = 2048,
        max_iter: int = int(1e5),
        confidence: float = 0.99,
        max_lo_iters: int = 5,
        inl_th: float = 2.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.inl_th_sq = inl_th ** 2

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

    def validate_inputs(self, X: torch.Tensor, x: torch.Tensor, minimal_sample_size: int) -> None:

        dim1 = X.shape[-1]
        assert dim1 in [3, 4]
        if dim1 == 4:
            assert torch.all(1.0 == X[:, 3])

        dim2 = x.shape[-1]
        assert dim2 in [2, 3]
        if dim2 == 3:
            assert torch.all(1.0 == x[:, 2])

        assert X.shape[0] == x.shape[0]
        assert X.shape[0] >= minimal_sample_size

    def forward(self, X: torch.Tensor, x: torch.Tensor, min_solver: BaseP3PSolver) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm. Device inferred now from the data.

        Args:
            x: 2D corrs :math:`(N, 2)`. Assumed to be already normalized by K.inv(): torch.Tensor
            X: 3D corrs :math:`(N, 3)`.

        Returns:
            - Estimated model, shape of (:math:`(3, 3)`, :math:`(3)`).
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
        """
        minimal_sample_size = min_solver.min_sample_size()
        self.validate_inputs(X, x, minimal_sample_size)

        #print(f"solving for {len(X)} points")
        X = P3P_RANSAC.to_homogeneous(X, 4)
        x = P3P_RANSAC.to_homogeneous(x, 3)
        best_score: float = sys.float_info.max
        num_tc: int = len(x)
        best_model_Rt = torch.zeros(3, 4, dtype=x.dtype, device=x.device)
        best_inliers: torch.Tensor = torch.zeros(num_tc, 1, device=x.device, dtype=torch.bool)

        for i in range(self.max_iter):

            # Sample minimal samples in batch to estimate models
            idxs = self.sample(minimal_sample_size, num_tc, self.batch_size, x.device)
            pts_2d_sampled = x[idxs]
            pts_3d_sampled = X[idxs]

            start = time.time()
            models, samples_X, samples_x = min_solver.solve(pts_3d_sampled, pts_2d_sampled)
            end = time.time()
            self.elapsed_time_in_solver += end - start
            if len(models) == 0:
                continue

            # Score the models and select the best one
            # TODO change the design
            if isinstance(min_solver, P3P_depth_solver):
                model, inliers, model_score = min_solver.score_models(X, x, models, samples_X, samples_x)
            else:
                model, inliers, model_score = self.score_models(X, x, models)

            # Store far-the-best model
            if model_score < best_score:

                in_ratio = inliers.sum() / inliers.shape[0]
                print(f"inlier ratio when updating: {in_ratio * 100}%")

                # TODO no local optimization
                # Store the best model
                best_model_Rt = model
                best_inliers = inliers
                best_score = model_score

                # Should we already stop?
                # TODO confirm the best score is actually the number of the inlers,
                # and not the continuous number (which should be considered when picking the best model?)
                new_max_iter = int(
                    self.max_samples_by_conf(best_inliers.sum().item(), num_tc, minimal_sample_size, self.confidence)
                )
                # Stop estimation, if the model is very good
                if (i + 1) * self.batch_size >= new_max_iter:
                    break
            print(i)

        #print(f"{i} iterations for {self.batch_size=}")
        return best_model_Rt, best_inliers


if __name__ == "__main__":

    min_solver = BaseP3PSolver(cv_flag=cv.SOLVEPNP_P3P)
    ransac = P3P_RANSAC(minimal_solver=min_solver, batch_size=1)
    # TODO test on something?
    # ransac.forward(None, None)
