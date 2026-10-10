"""Exercise the modified rasterizer and CUDA dependencies on one visible GPU.

Run from the repository root after installation:
    CUDA_VISIBLE_DEVICES=5 python scripts/check_cuda_extensions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import diff_gaussian_rasterization as raster
import sim3solve
import dpretrieval
from simple_knn._C import distCUDA2
from torch_scatter import scatter_max
from xformers.ops import memory_efficient_attention
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2


def main():
    assert torch.cuda.is_available(), "A visible NVIDIA GPU is required"
    points = torch.tensor([[0., 0., 2.], [.15, 0., 2.], [0., .15, 2.], [.15, .15, 2.]], device="cuda")
    distances = distCUDA2(points)
    assert distances.shape == (4,) and torch.isfinite(distances).all() and (distances > 0).all()
    values, _ = scatter_max(torch.tensor([1., 3., 2.], device="cuda"), torch.tensor([0, 0, 1], device="cuda"))
    assert values.tolist() == [3., 2.]
    q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.float16)
    attention = memory_efficient_attention(q, q, q)
    assert attention.shape == q.shape and torch.isfinite(attention).all()
    jacobian = torch.eye(7, device="cuda")[None].contiguous()
    delta, = sim3solve.solve_system(
        jacobian, -jacobian, torch.tensor([0], device="cuda"),
        torch.tensor([1], device="cuda"), torch.ones((1, 7), device="cuda"), .1, 0., -1,
    )
    expected = torch.tensor([-1. / 2.1, 1. / 2.1], device="cuda")[:, None].expand(2, 7)
    torch.testing.assert_close(delta, expected)
    projection = getProjectionMatrix2(.01, 100., 16., 16., 32., 32., 32, 32).T.contiguous().cuda()
    outputs = []
    for debug in (False, True):
        settings = raster.GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=.5, tanfovy=.5,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
            viewmatrix=torch.eye(4, device="cuda"), projmatrix=projection,
            projmatrix_raw=projection, sh_degree=0, campos=torch.zeros(3, device="cuda"),
            prefiltered=False, debug=debug, get_flag=True,
            metric_map=torch.ones(32 * 32, dtype=torch.int32, device="cuda"),
        )
        xyz = points.clone().requires_grad_()
        uv = torch.zeros_like(xyz, requires_grad=True)
        colors = torch.full((4, 3), .5, device="cuda", requires_grad=True)
        opacity = torch.full((4, 1), .7, device="cuda", requires_grad=True)
        scales = torch.full((4, 3), .05, device="cuda", requires_grad=True)
        rotations = torch.tensor([[1., 0., 0., 0.]] * 4, device="cuda", requires_grad=True)
        theta = torch.zeros((1, 3), device="cuda", requires_grad=True)
        rho = torch.zeros((1, 3), device="cuda", requires_grad=True)
        out = raster.GaussianRasterizer(settings)(
            means3D=xyz, means2D=uv, colors_precomp=colors, opacities=opacity,
            scales=scales, rotations=rotations, theta=theta, rho=rho,
        )
        assert len(out) == 6, "The legacy five-output renderer is incompatible"
        assert out[0].shape == (3, 32, 32) and (out[1] > 0).any()
        assert all(torch.isfinite(t).all() for t in out)
        (out[0].mean() + out[2].mean()).backward()
        for t in (xyz, uv, colors, opacity, scales, rotations, theta, rho):
            assert t.grad is not None and torch.isfinite(t.grad).all()
        assert colors.grad.abs().sum() > 0
        outputs.append([t.detach().clone() for t in out])
    for normal, debug in zip(*outputs):
        torch.testing.assert_close(normal, debug)
    torch.cuda.synchronize()
    print(json.dumps(dict(status="passed", torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), rasterizer=raster.__file__,
                         sim3solve=sim3solve.__file__, dpretrieval=dpretrieval.__file__), indent=2))


if __name__ == "__main__":
    main()
