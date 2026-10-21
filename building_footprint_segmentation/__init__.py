import logging

logger = logging.getLogger("segmentation")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)

info_file_handler = logging.FileHandler("../info.log", mode="a", encoding="utf-8")
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(formatter)

debug_file_handler = logging.FileHandler("../debug.log", mode="a", encoding="utf-8")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(formatter)

error_file_handler = logging.FileHandler("../error.log", mode="a", encoding="utf-8")
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

logger.addHandler(info_file_handler)
logger.addHandler(debug_file_handler)
logger.addHandler(error_file_handler)
