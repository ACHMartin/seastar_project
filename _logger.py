import logging
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the script with optional verbose mode.")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode (INFO logs)")
args, unknown = parser.parse_known_args()

# Logger configuration
logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,  # Set level based on --verbose
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs in the console
        logging.FileHandler("seastar_logger.log", mode="w")  # Save logs to a file and overwrite it each time we launch the code
    ]
)

# Create the logger
logger = logging.getLogger("seastar")  # Logger name based on your package
