import os
import logging
import json
import torch
from transformers import pipeline # Import the pipeline function

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables ---
sentiment_pipeline = None

# --- Initialization Function ---
def init():
    global sentiment_pipeline

    # --- Configuration ---
    # Use a standard DistilBERT model fine-tuned for sentiment analysis (outputs POSITIVE/NEGATIVE)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # We'll use CPU explicitly for this example
    device_id = -1 # -1 forces CPU for Hugging Face pipeline
    # Alternative if you want torch device object:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting initialization for sentiment analysis pipeline.")
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using device ID: {device_id} (CPU)")

    try:
        # --- Load Pipeline ---
        # The pipeline handles tokenization, model loading, inference, and post-processing (softmax, label mapping)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name, # Explicitly specify tokenizer
            device=device_id      # Use device_id for pipeline device mapping
            # device=device # Alternative using torch device object
        )
        logger.info("Sentiment analysis pipeline loaded successfully.")
        logger.info("Initialization complete.")

    except Exception as e:
        logger.error(f"Error during pipeline initialization: {e}", exc_info=True)
        # Raise the error to signal failure to Azure ML
        raise

# --- Run Function ---
def run(raw_data):
    global sentiment_pipeline
    logger.info("Received scoring request for sentiment analysis")

    # --- Input Validation and Parsing ---
    try:
        data = json.loads(raw_data)
        # Expect JSON like {"text": "input sentence"} or {"texts": ["sentence 1", "sentence 2"]}
        if "text" in data:
            input_texts = [data["text"]]
        elif "texts" in data:
            input_texts = data["texts"]
        else:
             raise ValueError("Input JSON must contain 'text' (string) or 'texts' (list of strings) field.")

        if not input_texts or not isinstance(input_texts, list) or not all(isinstance(t, str) for t in input_texts):
            raise ValueError("'text' or 'texts' must be a non-empty string or list of strings.")

        logger.info(f"Processing {len(input_texts)} input texts.")

    except Exception as e:
        logger.error(f"Error during input parsing: {e}", exc_info=True)
        return json.dumps({"error": f"Input parsing error: {type(e).__name__} - {e}"})

    # --- Run Inference via Pipeline ---
    try:
        logger.info("Running sentiment analysis pipeline...")
        # The pipeline automatically handles tokenization, inference, and applying softmax/labels
        # It can process a single string or a list of strings
        results = sentiment_pipeline(input_texts)
        logger.info(f"Pipeline processing complete. Results count: {len(results)}")
        # Example result: [{'label': 'POSITIVE', 'score': 0.9998}, {'label': 'NEGATIVE', 'score': 0.9985}]

        # --- Format and Return Output ---
        # Return the list of sentiment results directly
        # NOTE: This model gives 'POSITIVE' or 'NEGATIVE'. A different model would be needed
        # for 'neutral'. The output structure would likely be the same.
        return json.dumps({"sentiments": results})

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        return json.dumps({"error": f"Pipeline execution error: {type(e).__name__}"})

# --- Optional: Local Testing Placeholder ---
if __name__ == "__main__":
     # Example local test (requires transformers, torch, and potentially sentencepiece installed)
     try:
         print("Running local init...")
         init() # Load pipeline
         print("Init complete.")

         test_data_single = json.dumps({"text": "This is a wonderful library!"})
         print(f"\nTesting single input: {test_data_single}")
         result_single = run(test_data_single)
         print("Local test result (single):", result_single)

         test_data_batch = json.dumps({"texts": ["Azure ML is quite powerful.", "Deployment can sometimes be tricky."]})
         print(f"\nTesting batch input: {test_data_batch}")
         result_batch = run(test_data_batch)
         print("Local test result (batch):", result_batch)

     except Exception as e:
         print(f"\nLocal test failed: {e}", exc_info=True)