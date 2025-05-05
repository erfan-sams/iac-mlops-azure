# Azure MLOps NLP API Deployment

## Overview

This project documents the end-to-end process of deploying a Natural Language Processing (NLP) model as a scalable REST API on Microsoft Azure, leveraging MLOps best practices. The primary goal was to deploy a modern open-source Large Language Model (LLM) like Google's Gemma, intended eventually to power a Telegram bot.

Due to cloud subscription resource constraints encountered during development (GPU quotas and CPU instance OS Disk space limitations for the target LLM), the project currently features a successful deployment of a smaller transformer model (distilbert-base-uncased-finetuned-sst-2-english for sentiment analysis) to validate the MLOps pipeline and infrastructure. The original code (src/score_gemma_cpu.py) for the LLM deployment attempt is retained to demonstrate the intended architecture and challenges faced.

This repository showcases the journey, the MLOps techniques applied, the infrastructure setup, and the practical problem-solving involved in deploying ML models on the cloud.

## Key Features & MLOps Practices Demonstrated

*   **Infrastructure as Code (IaC):** Core Azure resources (Resource Group, ML Workspace, supporting services) managed via OpenTofu.
*   **Managed Online Endpoint:** Deployment target using Azure ML for scalable, managed real-time inference with auto-scaling capabilities (defaulting to minimum 1 instance).
*   **Environment Management:** Explored both complex custom environment builds (Dockerfile/Conda for the LLM) and a hybrid approach using a base image with Conda dependencies for the successful DistilBERT deployment.
*   **API Serving:** Created a functional REST API endpoint for NLP tasks.
*   **Cloud Deployment:** Hands-on experience deploying and debugging on Azure.
*   **Adaptability:** Navigated resource limitations (GPU quota, CPU disk space) and adapted the deployed model while maintaining the MLOps framework.

## Tech Stack

*   **Cloud Platform:** Microsoft Azure (Azure Machine Learning, Azure Blob Storage, Azure Container Registry, Key Vault, Application Insights, Log Analytics)
*   **Infrastructure as Code:** OpenTofu (Terraform Fork) managing core Azure resources.
*   **ML Model:** Target: Gemma 1B (LLM). Current Deployment: DistilBERT.
*   **ML Frameworks:** Transformers, PyTorch, BitsAndBytes (for GPU quantization attempt)
*   **Deployment Target:** Azure ML Managed Online Endpoint (GPU Instance). Current Deployment: CPU Instance.
*   **Environment:** Custom Docker/Conda Environments and Azure ML Curated Environments.
*   **Orchestration/CLI:** Azure CLI (az ml extension)
*   **Language:** Python

## Target LLM Architecture (Gemma)

The initial goal was to deploy the google/gemma-1b-it model using GPU acceleration and quantization for efficient inference.

**Phase 1: GPU Deployment Attempt (Blocked by Quota)**

1.  **Infrastructure:** Provisioned via OpenTofu as described below.
2.  **Model Storage:** Gemma model files downloaded and stored in Azure Blob Storage, registered as an Azure ML Model Asset.
3.  **Compute Target:** Intended to use a GPU-enabled instance (e.g., Standard_NC4as_T4_v3 or similar) available on Azure ML.
4.  **Environment:** A **custom environment was built using a Dockerfile** (environment/Dockerfile) and registered in Azure ML.
    *   The Dockerfile started from a base **acpt-pytorch-2.2-cuda12.1 image**.
    *   It used **Conda and/or Pip** to install specific versions of Python, PyTorch (GPU-enabled), Transformers, Accelerate, and **bitsandbytes** for quantization. 
    <!-- ^ Indentation of sub-bullets looks correct (usually 4 spaces for the `*`) -->
5.  **Scoring Logic (src/score_gemma_gpu.py):**
    *   `init()`: Configured BitsAndBytesConfig for 4-bit quantization (nf4, bfloat16 compute dtype). Loaded the model using `AutoModelForCausalLM.from_pretrained()` with the `quantization_config` and `device_map="auto"` to distribute across available GPUs. Model loaded from the mounted path (`AZUREML_MODEL_DIR`). 
    <!-- ^ Using backticks for function names/params improves readability. Ensure multi-line text under sub-bullets maintains indentation if needed, though here it reads as one block. -->
    *   `run()`: Handled chat-formatted input, tokenized inputs, and moved them to the model's device (`inputs.to(model.device)`), ran `model.generate()`, decoded output.
6.  **Deployment Config (deployment_gemma_gpu.yml):** Would have linked the **custom-built GPU environment** (referenced via `environment: azureml:<custom-gpu-env-name>:<version>`), the model asset, code, and GPU instance type.
7.  **Outcome:** This deployment **could not be executed** due to encountering zero quota for the required GPU VM SKUs on the Azure subscription.

**Phase 2: CPU Deployment Attempt (Blocked by Disk Space)**

1.  **Compute Target:** Pivoted to CPU instances `Standard_F4s_v2` which has a **32GB OS disk**.
2.  **Environment:** Custom environment built using Dockerfile/Conda for CPU, including PyTorch, Transformers, and other dependencies. The resulting container image size was substantial.
3.  **Scoring Logic (src/score_gemma_cpu.py):** Configured to load the model using `torch.float32` for CPU execution.
4.  **Deployment Config (deployment_gemma_cpu.yml):** Updated to use the CPU environment and a 8GB RAM / 32GB OS Disk CPU instance type.
5.  **Outcome:** Deployment attempts consistently failed, getting stuck in the "Creating" state or failing with errors indicative of **disk space exhaustion**. The combined size of the large custom container environment (including base OS, Python, ML libraries) and potentially temporary files used during container startup exceeded the **32GB OS disk capacity** available on the attempted instances. **Quota limitations prevented selecting alternative CPU instances with larger OS disks**, making this path unviable under the existing constraints.

## Current Deployment Architecture (DistilBERT - Working)

The successful deployment pivoted to using the `distilbert-base-uncased-finetuned-sst-2-english` model for sentiment analysis, leveraging a hybrid environment approach.

1.  **Infrastructure:** Provisioned via OpenTofu (as described below).
2.  **Environment Definition:** Defined via the `deployment.yml` configuration using:
    *   A base **Docker Image:** `mcr.microsoft.com/azureml/curated/lightgbm-3.3:61` specified via the `image:` key.
    *   A **Conda Environment File:** (`environment/conda_dependencies.yaml`) specified via the `conda_file:` key. Azure ML creates and activates this Conda environment on top of the base image at runtime.
3.  **Key Dependencies (conda_dependencies.yaml):** This file installs `python=3.10`, `pip`, and then uses `pip` to install:
    *   `torch==2.2.0`
    *   `transformers==4.51.3`
    *   `numpy~=1.26.4` - **Crucially, NumPy had to be pinned to a 1.x version** to resolve runtime errors encountered with default (likely 2.x) versions.
4.  **Scoring Logic (src/score_distilbert.py):**
    *   `init()`: Loads a sentiment-analysis pipeline from `transformers`, specifying the DistilBERT model. Model/tokenizer loaded directly from Hub. Runs within the activated Conda environment.
    *   `run()`: Receives text input, passes it to the pipeline, returns sentiment labels/scores.
5.  **Deployment Config (deployment.yml):** Links the base image (`image:` key) and the Conda file (`conda_file:` key), scoring code, CPU instance type (`Standard_F4s_v2`). Does *not* require a registered model asset.

## Infrastructure as Code (IaC) with OpenTofu

This project utilizes **OpenTofu** (an open-source fork of Terraform) to define, provision, and manage the core Azure infrastructure components required for the Azure Machine Learning workspace and its dependencies. This ensures the infrastructure setup is:

*   **Declarative:** Defined in version-controlled `.tofu` files.
*   **Repeatable:** Can be consistently provisioned across environments.
*   **Automated:** Reduces manual setup errors.

<!-- Add blank line before the list -->
The OpenTofu configuration (`infra/main.tofu`) manages the following:

1.  **Azure Resource Group (`azurerm_resource_group`):** The top-level container for all project resources.
2.  **Azure Key Vault (`azurerm_key_vault`):** Created and managed by Tofu for storing secrets.
3.  **Azure Log Analytics Workspace (`azurerm_log_analytics_workspace`):** Created and managed by Tofu, serving as the backend for Application Insights logs.
4.  **Azure Application Insights (`azurerm_application_insights`):** Created and managed by Tofu for monitoring the ML workspace and deployed endpoints. Explicitly linked to the Tofu-managed Log Analytics Workspace.
5.  **Azure Container Registry (`azurerm_container_registry`):** Created and managed by Tofu to store custom Docker images built by Azure ML (e.g., for custom environments).
6.  **Azure Machine Learning Workspace (`azurerm_machine_learning_workspace`):** The central hub, created and managed by Tofu. It is configured with:
    *   A **System-Assigned Managed Identity**.
    *   Links to the **Key Vault, Application Insights, and Container Registry** resources created above.
    *   A link to an **existing Azure Storage Account** (`llmbot4278888529`) referenced via a data `"azurerm_storage_account"` block. This demonstrates integrating Tofu-managed resources with pre-existing infrastructure.

**Dependency Management:** Explicit `depends_on` attributes and resource ID references are used throughout the configuration to ensure resources are created in the correct order (e.g., Log Analytics before Application Insights, all dependencies before the ML Workspace).

**Note:** While Tofu manages the core workspace infrastructure, the deployment of specific Azure ML assets like Endpoints, Deployments, Environments, and Models was handled using the Azure CLI (`az ml ...`) due to potential provider limitations or for finer-grained control over the ML-specific lifecycle.
<!-- ^ Add blank line before this Note paragraph -->

## Challenges & Learnings During Development

*   **GPU Quota Limits:** Blocked the initial Gemma GPU deployment.
*   **Disk Space Constraints (LLM Environment on CPU):** Blocked the Gemma CPU deployment due to the 32GB OS disk limit being exceeded by the large custom environment.
*   **Hybrid Environment Debugging (image + conda_file):** Successfully deployed DistilBERT using a base image (lightgbm) combined with a `conda_file`. However, debugging runtime errors required careful dependency management.
*   **NumPy Version Conflict:** Encountered runtime errors (Numpy is not available initially reported, likely masking the true issue) that were resolved by explicitly **pinning NumPy to a 1.x version** in the `conda_dependencies.yaml`. This highlights potential incompatibilities between newer library versions (like NumPy 2.x) and other components in the ML stack or base image.
*   **Custom Environment Alternatives:** While the hybrid approach worked, building fully custom environments (via `Dockerfile` build:) remains a viable, often more explicit, alternative for managing complex dependencies.
*   **Debugging Resource Exhaustion:** Difficulty obtaining logs when deployments fail due to resource limits (Disk Full) remains a challenge.
*   **Pivoting:** Adapting the deployed model was key to validating the MLOps pipeline within resource constraints.

## Future Work

*   **Implement Telegram Bot:** Develop a Python Telegram bot (`bot.py`) that interacts with the deployed API endpoint (currently the DistilBERT sentiment model, or the Gemma LLM if successfully deployed) to provide an interactive user interface.
*   **Explore Scale-to-Zero:** Configure `min_instances: 0` in the `deployment.yml`'s `scale_settings` for the online deployment to further minimize compute costs during inactive periods, accepting the trade-off of potential cold-start latency.
*   **Refine Monitoring:** Integrate deeper monitoring and alerting using Azure Monitor (Application Insights, Log Analytics) to track performance, errors, and costs more effectively.

## Setup & Deployment (Current DistilBERT Example)

1.  **Prerequisites:** Azure Subscription, Azure CLI, OpenTofu, Python.
2.  **Clone Repository:** `git clone <repo-url>`
3.  **Configure Azure:** `az login`, `az account set ...`
4.  **Prepare Conda Dependencies:** Ensure `environment/conda_dependencies.yaml` exists and **specifies a NumPy 1.x version** (e.g., `numpy~=1.26`, `numpy==1.26.4`).
5.  **Provision Infrastructure:** 
    <!-- Add blank line before code block and indent it -->
    ```bash
    cd infra && tofu init && tofu apply -auto-approve
    ```
6.  **Deploy Endpoint & Deployment:**
    *   Ensure `./aml-configs/deployment.yml` references the correct *DistilBERT* scoring script (e.g., `src/score_distilbert.py`).
    *   Ensure the `environment:` block correctly specifies **both** the `image: mcr.microsoft.com/azureml/curated/lightgbm-3.3:61` (or tag) **and** the `conda_file: ../environment/conda_dependencies.yaml`.
    *   Ensure the `model:` section is removed/commented out. Use an appropriate CPU `instance_type` (e.g., `Standard_F4s_v2`).
    <!-- Add blank lines around code blocks and indent them under the list item -->

    ```bash
    az ml online-endpoint create --file ./aml-configs/endpoint.yml -g <rg> -w <ws>
    ```

    ```bash
    az ml online-deployment create --name distilbert-deploy --endpoint <endpoint-name> --file ./aml-configs/deployment.yml --all-traffic -g <rg> -w <ws>
    ```
7.  **Monitor Deployment:** Check status and logs in Azure ML Studio.
8.  **Test Endpoint:** Use Studio Test Tab, `curl`, or Python requests (e.g., `{"text": "This is great!"}`).

## Cleanup

To avoid ongoing charges, remove the created Azure resources after completing the project.

1.  **Delete Azure ML Endpoint:** Remove the deployed endpoint and its associated compute resources first using the Azure CLI.
    <!-- Add blank lines around code block and indent it -->
    ```bash
    az ml online-endpoint delete --name <endpoint-name> -g <rg> -w <ws> --yes --no-wait
    ```
    *(Replace `<endpoint-name>`, `<rg>`, `<ws>` with your actual values)*

2.  **Archive Custom Environments (If created):** If you created and registered custom environments during experiments, archive them.
    <!-- Add blank lines around code block and indent it -->
    ```bash
    az ml environment archive --name <custom-env-name> --version <version> -g <rg> -w <ws>
    ```

3.  **Destroy Tofu-Managed Infrastructure:** Remove the core infrastructure components managed by OpenTofu.
    *   **Tofu CLI:** Navigate to the `infra/` directory in your terminal and run:
        <!-- Add blank lines around code block and indent it under the sub-bullet -->
        ```bash
        tofu destroy -auto-approve
        ```
        This command will use your Tofu state file to identify and delete the Resource Group, ML Workspace, Key Vault, ACR, App Insights, and Log Analytics workspace managed by Tofu. 
        <!-- Ensure this follow-up text is also indented correctly under the sub-bullet -->
