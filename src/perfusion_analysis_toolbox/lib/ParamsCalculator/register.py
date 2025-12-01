import os
import tempfile
import subprocess
import itk
import numpy as np

def register_and_select_slice(fixed_image_path: str, cta_np: np.ndarray, fixed_slice_index: int = 2) -> int:
    import numpy as np
    import tempfile
    import os
    import subprocess
    import itk

    assert cta_np.ndim == 3, "CTA image must be 3D (z, y, x)"

    with tempfile.TemporaryDirectory() as tmpdir:
        cta_path = os.path.join(tmpdir, "cta.nii.gz")
        transform_out = os.path.join(tmpdir, "trans.hdf5")
        warped_out = os.path.join(tmpdir, "warped.nii.gz")

        itk_img = itk.image_view_from_array(cta_np.astype(np.float32))
        itk.imwrite(itk_img, cta_path)

        cmd = [
            "unicarl-register",
            "--fixed", fixed_image_path,
            "--moving", cta_path,
            "--transform_out", transform_out,
            "--warped_moving_out", warped_out,
            "--io_iterations", "50"
        ]

        print("Running uniGradICON registration on CUDA...")
        print("Command:", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            print("Registration failed!")
            print(result.stderr)
            raise RuntimeError("Registration failed.")

        fixed_img = itk.imread(fixed_image_path, itk.F)
        moving_img = itk.imread(cta_path, itk.F)
        transform = itk.transformread(transform_out)[0]

        fixed_size = fixed_img.GetLargestPossibleRegion().GetSize()
        moving_size = moving_img.GetLargestPossibleRegion().GetSize()

        fixed_index = [fixed_size[0] // 2, fixed_size[1] // 2, fixed_slice_index]
        fixed_point = fixed_img.TransformIndexToPhysicalPoint(fixed_index)
        moving_point = transform.TransformPoint(fixed_point)
        moving_index = moving_img.TransformPhysicalPointToIndex(moving_point)
        
        print(f"Transformed index in moving image: {moving_index}")

        if not (0 <= moving_index[2] < moving_size[2]):
            raise ValueError(f"Mapped slice {moving_index[2]} is out of bounds for moving image with {moving_size[2]} slices.")

        print(f"Mapped fixed slice {fixed_slice_index} → moving slice {moving_index[2]}")
        print("Fixed image direction:\n", np.array(fixed_img.GetDirection()).reshape(3, 3))
        print("Moving image direction:\n", np.array(moving_img.GetDirection()).reshape(3, 3))

        return moving_index[2]

