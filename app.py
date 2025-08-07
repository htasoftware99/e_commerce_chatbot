import os
import sys
import logging
from datetime import datetime
from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from dotenv import load_dotenv

from src.data_ingestion import DataIngestor
from src.rag_chain import RAGChainBuilder

load_dotenv()

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("app_logger")

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
        return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"

    def __str__(self):
        return self.error_message

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")

def create_app():
    app = Flask(__name__)

    try:
        logger.info("Starting data ingestion...")
        vector_store = DataIngestor().ingest(load_existing=True)
        logger.info("Data ingestion complete.")

        rag_chain = RAGChainBuilder(vector_store).build_chain()
        logger.info("RAG chain initialized.")
    except Exception as e:
        logger.error("Error during app initialization.")
        raise CustomException("Initialization failed", e)

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        try:
            user_input = request.form["msg"]
            logger.info(f"Received user input: {user_input}")

            response = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user-session"}}
            )["answer"]

            logger.info(f"Bot response: {response}")
            return response

        except Exception as e:
            logger.error("Error in /get endpoint")
            raise CustomException("Error while generating response", e)

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)