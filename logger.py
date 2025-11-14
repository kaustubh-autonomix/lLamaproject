import logging

logger = logging.getLogger("lLamaproject")
logger.setLevel(logging.INFO)


handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
