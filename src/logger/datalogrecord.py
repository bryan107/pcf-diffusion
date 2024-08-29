import logging

import numpy as np
import numpy.typing
import torch


class DataLogRecord(logging.LogRecord):
    NUM_ELEMENTS_SHOWN = 2

    # The advantage of this logRecord is that we lazily transform the data for logging.
    # This is particularly useful when logs are called in computationally expensive areas of the code.
    # To use it, log the data with the following syntax:
    # logger.info("tensor %s", tensor)
    # do not use f-strings.
    def getMessage(self):
        msg = str(self.msg)
        with np.printoptions(precision=4, suppress=True, linewidth=115):
            if self.args:
                processed_args = []
                for arg in self.args:
                    if isinstance(arg, torch.Tensor):
                        processed_args.append(self._format_tensor(arg))
                    elif isinstance(arg, np.ndarray):
                        processed_args.append(self._format_array(arg))
                    else:
                        processed_args.append(arg)
                msg = msg % tuple(processed_args)
        return msg

    def _format_tensor(self, tensor: torch.Tensor):
        shape = list(tensor.shape)
        if len(shape) == 3:
            means: torch.Tensor = tensor.mean((0, 1))
            stds: torch.Tensor = tensor.std((0, 1))

            if shape[1] < 80:
                return (
                    "- shape: "
                    + str(shape)
                    + " with means (last dimension): "
                    + self._tensor2string(means)
                    + " and stds: "
                    + self._tensor2string(stds)
                    + f" with {DataLogRecord.NUM_ELEMENTS_SHOWN} first batch elements: \n"
                    + self._tensor2string(
                        tensor[: DataLogRecord.NUM_ELEMENTS_SHOWN, :, :].transpose(1, 2)
                    )
                )
            return (
                "- shape: "
                + str(shape)
                + " with means (last dimension): "
                + self._tensor2string(means)
                + " and stds: "
                + self._tensor2string(stds)
                + " with first batch element: \n"
                + self._tensor2string(tensor[0, :, :].transpose(0, 1))
            )
        else:
            return self._tensor2string(tensor)

    def _format_array(self, array: np.typing.ArrayLike):
        shape = list(array.shape)
        if len(shape) == 3:
            means: np.typing.ArrayLike = np.mean(array, axis=(0, 1))

            if shape[1] < 80:
                return (
                    "- shape: "
                    + str(shape)
                    + " with means (last dimension): "
                    + np.array2string(means)
                    + f" with {DataLogRecord.NUM_ELEMENTS_SHOWN} first batch elements: \n"
                    + np.array2string(
                        array[: DataLogRecord.NUM_ELEMENTS_SHOWN, :, :].transpose(
                            0, 2, 1
                        )
                    )
                )
            return (
                "- shape: "
                + str(shape)
                + " with means (last dimension): "
                + np.array2string(means)
                + " with first batch element: \n"
                + np.array2string(array[0, :, :].transpose(0, 1))
            )
        else:
            return np.array2string(array)

    def _tensor2string(self, tensor):
        return np.array2string(tensor.detach().cpu().numpy())


if __name__ == "__main__":
    import logging.config

    from src.logger.init_logger import set_config_logging

    set_config_logging()
    logger = logging.getLogger(__name__)

    tensor = torch.rand(2, 10, 3)
    logger.info("tensor %s", tensor)
    logger.info("array %s", tensor.numpy())
