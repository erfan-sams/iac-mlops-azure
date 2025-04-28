# infra/main.tofu

# --- Core Resources ---

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# --- Workspace Dependencies (Explicit Definition) ---

# Use a DATA SOURCE to reference the EXISTING Storage Account
data "azurerm_storage_account" "sa" {
  # Provide the name and resource group of the existing storage account
  name                = var.storage_account_name # Ensure variable holds "llmbot4278888529"
  resource_group_name = var.resource_group_name
  depends_on          = [azurerm_resource_group.rg] # Ensure RG exists before lookup
}

# Data block to get tenant ID needed for Key Vault
data "azurerm_client_config" "current" {}

# Key Vault (Tofu Creates New)
resource "azurerm_key_vault" "kv" {
  name                       = var.key_vault_name # Ensure variable has a unique name
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  tags                       = var.tags
  depends_on                 = [azurerm_resource_group.rg]
}

# Log Analytics Workspace (Dependency for modern App Insights)
resource "azurerm_log_analytics_workspace" "log" {
  name                = var.log_analytics_workspace_name # Ensure variable has a suitable name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018" # Common SKU
  retention_in_days   = 30
  tags                = var.tags
  depends_on          = [azurerm_resource_group.rg]
}


# Application Insights (Tofu Creates New / Manages Link)
resource "azurerm_application_insights" "ai" {
  name                = var.app_insights_name # Ensure variable has a suitable name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
  # Explicitly link to the Log Analytics Workspace managed by Tofu
  workspace_id = azurerm_log_analytics_workspace.log.id
  tags         = var.tags
  # Ensure Log Analytics workspace is created first
  depends_on = [azurerm_log_analytics_workspace.log]
}

# Container Registry (Tofu Creates New)
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name # Ensure variable has a unique name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
  admin_enabled       = false
  tags                = var.tags
  depends_on          = [azurerm_resource_group.rg]
}

# --- Azure ML Workspace ---
resource "azurerm_machine_learning_workspace" "mlw" {
  name                = var.workspace_name # e.g., "llm-bot"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  # Link dependencies: Existing Storage (data source), New KV, AI, ACR (resources)
  application_insights_id = azurerm_application_insights.ai.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = data.azurerm_storage_account.sa.id
  container_registry_id   = azurerm_container_registry.acr.id

  identity { type = "SystemAssigned" }
  sku_name                      = "Basic" # Or "Enterprise"
  public_network_access_enabled = true    # Adjust if network isolation needed
  tags                          = var.tags

  # Ensure dependencies are created/looked up before workspace
  depends_on = [
    data.azurerm_storage_account.sa,
    azurerm_key_vault.kv,
    azurerm_application_insights.ai, # Which itself depends on Log Analytics
    azurerm_container_registry.acr
  ]
}