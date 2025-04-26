import os
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables ---
model = None
tokenizer = None

# --- Initialization Function ---
def init():
    global model, tokenizer

    # --- Configuration ---
    # AZUREML_MODEL_DIR is an environment variable created by Azure ML
    # It points to the local path where the registered model files are mounted.
    model_path = os.getenv("AZUREML_MODEL_DIR", ".") # Default to current dir if env var not found
    compute_dtype = torch.bfloat16 

    logger.info(f"Starting initialization.")
    logger.info(f"Attempting to load model from path: {model_path}")
    logger.info(f"Using compute dtype: {compute_dtype}")

    if not os.path.exists(model_path) or not os.listdir(model_path):
         logger.error(f"Model directory ({model_path}) not found or empty. Check model mounting in deployment config.")
         raise FileNotFoundError(f"Model directory not found or empty: {model_path}")

    # --- bitsandbytes Configuration ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info("Using BitsAndBytes 4-bit quantization config on load.")
    quantization_config_param = bnb_config 
    logger.info("Loading model without explicit on-the-fly quantization config.")


    try:
        # --- Load Tokenizer ---
        logger.info(f"Loading tokenizer from path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully.")

        # --- Load Model ---
        logger.info(f"Loading model from path: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, # Load from the mounted Azure ML model path
            device_map="auto",
            torch_dtype=compute_dtype,
            quantization_config=quantization_config_param,
            # attn_implementation="flash_attention_2", # Optional
        )
        logger.info("Model loaded successfully.")

        model.eval()
        logger.info("Initialization complete. Model ready for inference.")

    except Exception as e:
        logger.error(f"Error during Azure ML initialization: {e}", exc_info=True)
        raise

# --- Run Function ---
def run(raw_data):
    global model, tokenizer
    logger.info(f"Received scoring request: {len(raw_data)} bytes")

    # --- Input Validation and Parsing ---
    try:
        data = json.loads(raw_data)
        messages = data.get("messages")
        parameters = data.get("parameters", {}) # Still accept parameters, but ignore sampling ones

        if not messages or not isinstance(messages, list):
            raise ValueError("Input JSON must contain a 'messages' field as a non-empty list.")
        logger.info(f"Received messages: {messages}")

    except Exception as e:
        logger.error(f"Error during input parsing: {e}", exc_info=True)
        # Use f-string for cleaner formatting
        return json.dumps({"error": f"Unexpected error parsing input: {type(e).__name__}"})

    # --- Prepare Inputs for Model ---
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        logger.info(f"Applied chat template. Input token length: {input_len}")

    except Exception as e:
        logger.error(f"Error applying chat template or tokenizing: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to process input messages: {type(e).__name__}. Check message format."})

    # --- Set Generation Parameters (Simplified for Greedy) ---
    try:
        # Always use greedy search (do_sample=False)
        # Only parameter we care about from input is max_new_tokens
        generation_kwargs = {
            "max_new_tokens": int(parameters.get("max_new_tokens", 150)), # Default 150
            "do_sample": False,
            # "pad_token_id": tokenizer.eos_token_id # Often not needed
        }
    except Exception as e:
        logger.error(f"Error processing generation parameters: {e}", exc_info=True)
        return json.dumps({"error": f"Invalid generation parameters: {type(e).__name__}"})

    # --- Run Inference ---
    try:
        with torch.inference_mode():
            logger.info(f"Generating response with parameters: {generation_kwargs}")
            outputs = model.generate(**inputs, **generation_kwargs)
            generation_ids = outputs[0][input_len:]
            logger.info(f"Generated {len(generation_ids)} new tokens.")

        # --- Decode Response ---
        decoded_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
        logger.info(f"Decoded response length: {len(decoded_text)}")

        # --- Format and Return Output ---
        return json.dumps({"generated_text": decoded_text})

    except Exception as e:
        logger.error(f"Error during model generation or decoding: {e}", exc_info=True)
        return json.dumps({"error": f"Error during model inference: {type(e).__name__}"})

# --- Optional: Local Testing Placeholder ---
if __name__ == "__main__":
     print("score.py loaded. Contains init() and run() for Azure ML deployment.")
     # Add basic parsing tests here if desired, model loading won't work locally easily.