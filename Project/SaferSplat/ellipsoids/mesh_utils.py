import numpy as np
import open3d as o3d
import tqdm

def create_gs_mesh(means, rotations, scalings, colors, res=4, transform=None, scale=None):
    scene = o3d.geometry.TriangleMesh()

    # Nerfstudio performs the transform first, then does the scaling
    if scale is not None:
        means = means * scale
        scalings = scalings * scale

    if transform is not None:
        rot = transform[:3, :3]
        t = transform[:3, -1]

        means = np.matmul(rot, means[..., None]).squeeze() + t

        rotations = np.matmul(rot, rotations)

    base_sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=res)

    colors = np.clip(colors, 0, 1)

    for (mean, R, S, col) in tqdm.tqdm(zip(means, rotations, scalings, colors), desc="Generating mesh"):
        primitive = o3d.geometry.TriangleMesh(base_sphere)
        points = np.asarray(primitive.vertices)
        points *= S[None]
        primitive.vertices = o3d.utility.Vector3dVector(points)
        primitive = primitive.paint_uniform_color(col)
        primitive = primitive.rotate(R)
        primitive = primitive.translate(mean)
        scene += primitive

    return scene