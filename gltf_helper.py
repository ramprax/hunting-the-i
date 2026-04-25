import numpy as np

import pygltflib


def make_gltf(point_list, triangle_list, scale_down_factor=1.0):
    points = np.array(
        point_list,
        dtype="float32",
    )/scale_down_factor

    triangles = np.array(
        triangle_list,
        dtype="uint16",
    )

    triangles_binary_blob = triangles.flatten().tobytes()
    points_binary_blob = points.tobytes()
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1), indices=0
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_binary_blob) + len(points_binary_blob)
            )
        ],
    )
    gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
    gltf.convert_buffers(pygltflib.BufferFormat.DATAURI)

    return gltf


def make_glb(point_list, triangle_list):
    gltf = make_gltf(point_list, triangle_list)
    glb = b"".join(gltf.save_to_bytes())  # save_to_bytes returns an array of the components of a glb
    return glb


def save_gltf_to_file(gltf, filepath):
    gltf.save(filepath)
