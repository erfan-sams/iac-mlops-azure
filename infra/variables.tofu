# infra/variables.tofu

variable "resource_group_name" {
  description = "Name of the Azure Resource Group"
  type        = string
  default     = "erfan.samieyansahneh-rg"
}

variable "location" {
  description = "Azure region for deployment"
  type        = string
  default     = "italynorth" # Or your preferred region
}

variable "workspace_name" {
  description = "Name of the Azure ML Workspace"
  type        = string
  default     = "llm-bot-v1"
}

variable "storage_account_name" {
  description = "Name of the EXISTING Storage Account to use"
  type        = string
  default     = "llmbot4278888529" # YOUR EXISTING STORAGE NAME
}

variable "key_vault_name" {
  description = "Globally unique name for the NEW Key Vault"
  type        = string
  default     = "kv-llmbot-erfan-2025" # EXAMPLE - MAKE UNIQUE
}

variable "log_analytics_workspace_name" {
  description = "Name for the NEW Log Analytics Workspace (unique within RG)"
  type        = string
  default     = "log-llmbot-erfan" # EXAMPLE - Choose name
}

variable "app_insights_name" {
  description = "Name for the NEW Application Insights (unique within RG)"
  type        = string
  default     = "ai-llmbot-erfan" # EXAMPLE - Choose name
}

variable "acr_name" {
  description = "Globally unique name for the NEW Container Registry"
  type        = string
  default     = "acrllmboterfan2025" # EXAMPLE - MAKE UNIQUE
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    Project   = "LLM-Deployment-Demo"
    Owner     = "erfan"
    ManagedBy = "OpenTofu"
  }
}