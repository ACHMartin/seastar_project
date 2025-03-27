import logging

# Logger configuration
logging.basicConfig(
    level=logging.INFO,  # Default level (can be DEBUG, WARNING, ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs in the console
        logging.FileHandler("seastar_logger.log", mode="w")  # Save logs to a file and overwrite it each time we launch the code
    ]
)

# Create the logger
logger = logging.getLogger("seastar")  # Logger name based on your package

# Optional: Adjust the level depending on the environment
logger.setLevel(logging.DEBUG)  # Enable in development mode