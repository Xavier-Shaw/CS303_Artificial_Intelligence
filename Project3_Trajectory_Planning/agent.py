import time
from typing import Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch
from functorch import vmap
import os.path as osp
from torch.autograd import Variable


#  Net
class NetModel(torch.nn.Module):
    def __init__(self, n_features, n_output):
        super(NetModel, self).__init__()
        self.hiddenLayer1 = torch.nn.Linear(n_features, 100)
        self.hiddenLayer2 = torch.nn.Linear(100, 50)
        self.hiddenLayer3 = torch.nn.Linear(50, 20)
        self.predictLayer = torch.nn.Linear(20, n_output)

    # 搭建神经网络， 输入data: x
    def forward(self, x):
        # 使用隐藏层加工x，用激励函数激活隐藏层输出的信息
        x = F.relu(self.hiddenLayer1(x))
        # 使用预测层预测
        x = self.hiddenLayer2(x)
        x = self.hiddenLayer3(x)
        x = self.predictLayer(x)
        return x


P = 3  # spline degree
N_CTPS = 5  # number of control points
RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256
SET_SIZE = 400


class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        self.nn = NetModel(256, 10)
        self.path = osp.join(osp.dirname(__file__), 'model1.pth')
        self.nn.load_state_dict(torch.load(self.path))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)

        # TODO: compute the firing speed and angle that would give the best score.
        # Example: return a random configuration
        start_time = time.time()
        classes = self.nn(target_features)
        predict = torch.tensor([c.cpu().detach().numpy() for c in classes])
        predict = numpy.argmax(predict, axis=1)

        def generate_batch(x, y, z):
            return x * y + z

        def c_traj(ct):
            t = torch.linspace(0, N_CTPS - P, 100, device=torch.device("cpu"))
            knots = torch.cat([
                torch.zeros(P, device=torch.device("cpu")),
                torch.arange(N_CTPS + 1 - P, device=torch.device("cpu")),
                torch.full((P,), N_CTPS - P, device=torch.device("cpu")),
            ])
            ctps = torch.cat([
                torch.tensor([[0., 0.]], device=torch.device("cpu")),
                ct,
                torch.tensor([[N_CTPS, 0.]], device=torch.device("cpu"))
            ])
            splv = self.splev(t, knots, ctps, P)
            cdist = torch.cdist(target_pos, splv)
            d = cdist.min(-1).values
            d = torch.where(d <= RADIUS, 1, RADIUS / d)
            score = torch.sum(d * class_scores[predict], dim=-1)
            # ct.requires_grad = True
            # print(ct.grad)
            # functorch.grad()
            # ct.data = ct.data + learning_rate * ct.grad / torch.norm(ct.grad)
            return score

        # loops = 0
        x = torch.randn((SET_SIZE, N_CTPS - 2, 2))
        y = torch.tensor([N_CTPS - 2, 2.])
        z = torch.tensor([1., -1.])
        result = functorch.vmap(generate_batch, in_dims=(0, None, None))(x, y, z)
        scores = vmap(c_traj)(result)
        idx = torch.argmax(scores)

        best_ctps = result[idx].detach().clone()
        best_score = scores[idx]

        while time.time() - start_time < 0.23:
            print(best_score)
            x = torch.randn((SET_SIZE, N_CTPS - 2, 2))
            y = torch.tensor([N_CTPS - 2, 2.])
            z = torch.tensor([1., -1.])

            result = functorch.vmap(generate_batch, in_dims=(0, None, None))(x, y, z)
            scores = vmap(c_traj)(result)
            idx = torch.argmax(scores)
            score = scores[idx]
            if score > best_score:
                best_score = score
                best_ctps = result[idx].detach().clone()

        ctps_inter = best_ctps.detach().clone()
        ctps_inter.requires_grad = True
        learning_rate = 0.3
        optimizer = torch.optim.Adam([ctps_inter], lr=learning_rate, maximize=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
        # print('----------------')
        epoch = 0
        while time.time() - start_time < 0.29:
            optimizer.zero_grad()

            score = self.evaluate(self.compute_traj(ctps_inter), target_pos, class_scores[predict], RADIUS)
            # print(score)
            score.backward()
            if score > best_score:
                best_score = score
                best_ctps = ctps_inter.detach().clone()
            # ctps_inter.data = ctps_inter.data + learning_rate * ctps_inter.grad / torch.norm(ctps_inter.grad)
            optimizer.step()
            scheduler.step()
            epoch += 1

            # if loops >= SET_SIZE - 1:
            #     break

            # x = torch.randn((SET_SIZE, N_CTPS - 2, 2))
            # y = torch.tensor([N_CTPS - 2, 2.])
            # z = torch.tensor([1., -1.])
            #
            # result = functorch.vmap(generate_batch, in_dims=(0, None, None))(x, y, z)
            # scores = vmap(c_traj)(result)
            # idx = torch.argmax(scores)
            # score = scores[idx]
            # if score > best_score:
            #     best_score = score
            #     best_ctps = result[idx].detach().clone()
            # loops += 1
            # score = self.evaluate(self.compute_traj(result[loops]), target_pos, class_scores[predict], RADIUS)
            # if score > best_score:
            #     best_ctps = result[loops]
            #     best_score = score

            # # result = vmap(model, in_dims=0)(result)
            # for data in result:
            #     # print(type(data))
            #     optimizer = torch.optim.Adam([data], lr=learning_rate)
            #     optimizer.zero_grad()
            #     score = self.evaluate(self.compute_traj(data), target_pos, class_scores[predict], RADIUS)
            #     score = score.float()
            #     score.requires_grad = True
            #     score.backward()
            #     optimizer.step()
            #     # scheduler.step(score)

        # # print(loops)
        # print('--------')
        # print('best_score:' + str(best_score))
        return best_ctps

    def compute_traj(self, ctps_inter: torch.Tensor):
        """Compute the discretized trajectory given the second to the second control points"""
        t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
        knots = torch.cat([
            torch.zeros(P, device=ctps_inter.device),
            torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
            torch.full((P,), N_CTPS - P, device=ctps_inter.device),
        ])
        ctps = torch.cat([
            torch.tensor([[0., 0.]], device=ctps_inter.device),
            ctps_inter,
            torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
        ])
        return self.splev(t, knots, ctps, P)

    def evaluate(self,
                 traj: torch.Tensor,
                 target_pos: torch.Tensor,
                 target_scores: torch.Tensor,
                 radius: float,
                 ) -> torch.Tensor:
        """Evaluate the trajectory and return the score it gets.

        Parameters
        ----------
        traj: Tensor of shape `(*, T, 2)`
            The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
        target_pos: Tensor of shape `(N, 2)`
            x-y positions of shape where `N` is the number of targets.
        target_scores: Tensor of shape `(N,)`
            Scores you get when the corresponding targets get hit.
        """
        cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
        d = cdist.min(-1).values
        hits = (d <= radius)
        d[hits] = 1
        d[~hits] = radius / d[~hits]
        hits.require_grad = True
        value = torch.sum(d * target_scores, dim=-1)
        return value

    def splev(self,
              x: torch.Tensor,
              knots: torch.Tensor,
              ctps: torch.Tensor,
              degree: int,
              der: int = 0
              ) -> torch.Tensor:
        """Evaluate a B-spline or its derivatives.

        See https://en.wikipedia.org/wiki/B-spline for more about B-Splines.
        This is a PyTorch implementation of https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

        Parameters
        ----------
        x : Tensor of shape `(t,)`
            An array of points at which to return the value of the smoothed
            spline or its derivatives.
        knots: Tensor of shape `(m,)`
            A B-Spline is a piece-wise polynomial.
            The values of x where the pieces of polynomial meet are known as knots.
        ctps: Tensor of shape `(n_ctps, dim)`
            Control points of the spline.
        degree: int
            Degree of the spline.
        der: int, optional
            The order of derivative of the spline to compute (must be less than
            or equal to k, the degree of the spline).
        """
        if der == 0:
            return self._splev_torch_impl(x, knots, ctps, degree)
        else:
            assert der <= degree, "The order of derivative to compute must be less than or equal to k."
            n = ctps.size(-2)
            ctps = (ctps[..., 1:, :] - ctps[..., :-1, :]) / (knots[degree + 1:degree + n] - knots[1:n]).unsqueeze(-1)
            return degree * self.splev(x, knots[..., 1:-1], ctps, degree - 1, der - 1)

    def _splev_torch_impl(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
        """
            x: (t,)
            t: (m, )
            c: (n_ctps, dim)
        """
        assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m= n + k + 1

        x = torch.atleast_1d(x)
        assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
        n = c.size(0)
        u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
        x = x.unsqueeze(-1)
        d = c[u - k + torch.arange(k + 1, device=c.device)].contiguous()
        for r in range(1, k + 1):
            j = torch.arange(r - 1, k, device=c.device) + 1
            t0 = t[j + u - k]
            t1 = t[j + u + 1 - r]
            alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
            d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
        return d[:, k]
