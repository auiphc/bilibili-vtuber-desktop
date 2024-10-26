import os
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage

from rembg.sessions.base import BaseSession


class DisCustomSession(BaseSession):
    """
    This class represents a custom session for isnet.
    """

    def __init__(
        self,
        model_name: str,
        sess_opts: ort.SessionOptions=None,
        providers=None,
        *args,
        **kwargs
    ):
        """
        Initialize a new DisCustomSession object.

        Parameters:
            model_name (str): The name of the model.
            sess_opts (ort.SessionOptions): The session options.
            providers: The providers.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If model_path is None.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path is required")
        
        if sess_opts is None:
            sess_opts = ort.SessionOptions()
            if "OMP_NUM_THREADS" in os.environ:
                sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
                sess_opts.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

        super().__init__(model_name, sess_opts, providers, *args, **kwargs)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Use a pre-trained model to predict the object in the given image.

        Parameters:
            img (PILImage): The input image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[PILImage]: A list of predicted mask images.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0), (1024, 1024)),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Download the model files.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The absolute path to the model files.
        """
        model_path = kwargs.get("model_path")
        if model_path is None:
            return

        return os.path.abspath(os.path.expanduser(model_path))

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Get the name of the pre-trained model.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The name of the pre-trained model.
        """
        return "isnet-custom"


if __name__ == "__main__":
    from configparser import ConfigParser
    from argparse import ArgumentParser
    from rembg import remove

    config = ConfigParser()
    config.read('config.ini')

    parser = ArgumentParser()
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    model_path = config.get('Settings', 'rembg_model_path', fallback=None)

    session = DisCustomSession("isnet-custom", model_path=model_path,
                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    image = Image.open(args.image)
    image = remove(image, session=session)
    image.show()