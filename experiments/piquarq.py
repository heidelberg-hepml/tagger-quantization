import math

import torch
from parq.optim.parq import amp_custom_fwd, channel_bucketize
from parq.optim.proxmap import ProxMap
from torch import Tensor


def compute_beta_exponential(
    t: int, t1: int, t2: int, beta_start: float = 1e-3, beta_end: float = 0.5
) -> float:
    """Compute beta using exponential schedule beta^(T) = 2^(alpha(T - T_0)).

    Maps iteration t in [t1, t2) to beta values using exponential growth.
    At t1, beta = beta_start (weak regularization).
    At t2, beta = beta_end (>= 0.5 for hard rounding).

    Args:
        t: Current iteration (should be in range [t1, t2))
        t1: Start of annealing window
        t2: End of annealing window (where beta reaches beta_end)
        beta_start: Initial beta value at t1
        beta_end: Beta value at t2

    Returns:
        beta: Regularization strength at iteration t
    """
    assert t >= t1 and t < t2, "Beta schedule: ensure t1 <= t < t2"

    # Map t to T in range [1, t2 - t1]
    T = t - t1 + 1
    T_max = t2 - t1

    alpha = (math.log2(beta_end) - math.log2(beta_start)) / (T_max - 1)

    T0 = 1 - math.log2(beta_start) / alpha

    # Compute beta for current T
    beta = 2.0 ** (alpha * (T - T0))

    return beta


class ProxPiQuaRQ(ProxMap):
    def __init__(
        self,
        anneal_start: int,
        anneal_end: int,
        beta_start: float = 1e-3,
        beta_end: float = 1.0,
    ) -> None:
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.quantize_bounds = True

    @torch.no_grad()
    @amp_custom_fwd(cast_inputs=torch.float32)
    def apply_(
        self,
        p: Tensor,
        q: Tensor,
        Q: Tensor,
        step_count: int,
        dim: int | None = None,
    ) -> float:
        """Prox-map based on piecewise quadratic regularization with gradual annealing.

        The parameter beta = tau * lambda represents the effective regularization strength.
        - beta = 0: identity function (no regularization)
        - 0 < beta < 0.5: soft rounding near integers, soft thresholding outside [-M, M]
        - beta = 0.5: hard rounding, hard thresholding

        Annealing: beta follows exponential schedule beta^(T) = 2^(α(T - T₀))
        - At anneal_start (t1): beta = beta_start (weak regularization, typically ~1e-3)
        - At anneal_end (t2): beta = beta_end (transition to hard rounding, typically 0.5)
        - After anneal_end: beta stays at beta_end or continues growing

        Args:
            p: Parameter tensor to quantize
            q: Previous quantized values
            Q: Quantization grid. If dim is None, Q is 1-D. If dim is specified, Q is 2-D (channels × grid_points)
            step_count: Current training step for annealing
            dim: If None, use scalar M (max of Q). If specified, use per-channel M (max per row of Q).
        """

        if step_count >= self.anneal_end:
            # After annealing: beta has reached beta_end
            beta = self.beta_end
            if q is None:
                # hard quantization to the nearest point in Q
                Q_mid = (Q[..., :-1] + Q[..., 1:]) / 2
                if dim is None:
                    q = Q[torch.bucketize(p, Q_mid)]
                else:
                    q = Q.gather(1, channel_bucketize(p, Q_mid))
                if not self.quantize_bounds:
                    raise NotImplementedError()
                    # beta = compute_beta_exponential(
                    #     step_count,
                    #     self.anneal_start,
                    #     self.anneal_end,
                    #     self.beta_start,
                    #     self.beta_end,
                    # )
                    # if dim is None:
                    #     # Q is 1-D: M is the maximum value
                    #     M = Q.max()
                    # else:
                    #     # Q is 2-D (channels × grid_points): M is max per channel
                    #     M = Q.max(dim=1, keepdim=False)  # Shape: (num_channels,)
                    #     # Reshape M to broadcast correctly along the specified dimension
                    #     M_shape = [1] * p.ndim
                    #     M_shape[dim] = -1
                    #     M = M.view(*M_shape)
                    # u = torch.abs(p)
                    # s = torch.sign(p)
                    # mask_M = u < M
                    # denom = 1 + beta
                    # ratio = (u - M - beta / 2) / denom
                    # case2 = torch.relu(ratio) + M
                    # q = torch.where(mask_M, q, s * case2)
            p.copy_(q)
        else:
            # Compute beta using exponential schedule via helper function
            if step_count < self.anneal_start:
                # Before annealing: use minimum beta (weak regularization)
                beta = self.beta_start
            else:
                # During annealing: compute beta based on current step
                beta = compute_beta_exponential(
                    step_count,
                    self.anneal_start,
                    self.anneal_end,
                    self.beta_start,
                    self.beta_end,
                )

            s = torch.sign(p)
            u = torch.abs(p)
            u_m = torch.floor(u)

            # Extract M from Q (max value in quantization grid)
            if dim is None:
                # Q is 1-D: M is the maximum value
                M = Q.max()
            else:
                # Q is 2-D (channels × grid_points): M is max per channel
                M = Q.max(dim=1, keepdim=False)  # Shape: (num_channels,)
                # Reshape M to broadcast correctly along the specified dimension
                M_shape = [1] * p.ndim
                M_shape[dim] = -1
                M = M.view(*M_shape)

            mask_M = u < M

            # Case 1: u < M
            # clip((t - beta / 2) / (1 - beta), 0, 1) + u_m, where t = u - u_m
            t_case1 = u - u_m
            denom = 1 - beta
            ratio = (t_case1 - beta / 2) / denom
            case1 = torch.clamp(ratio, 0, 1) + u_m

            # Case 2 : u >= M
            # relu((1 - beta) * t - beta / 2) + M, where t = u - M
            t_case2 = u - M

            if self.quantize_bounds:
                case2 = torch.relu((1 - beta) * t_case2 - beta / 2) + M
            else:
                raise NotImplementedError()
                # denom = 1 + beta
                # ratio = (t_case2 - beta / 2) / denom
                # case2 = torch.relu(ratio) + M

            q_abs = torch.where(mask_M, case1, case2)

            # Apply sign
            q_new = s * q_abs

            # In-place update of model parameters
            p.copy_(q_new)

        return 1 - beta
