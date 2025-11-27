import logging
import os
import sys
from datetime import datetime

# Step 1: Create log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Step 2: Create logs directory path
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Step 3: Full log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Step 4: Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info("Logging setup successful!")
