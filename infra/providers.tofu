# infra/providers.tofu

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
  # Optional: Configure backend for remote state storage (recommended for teams)
  # backend "azurerm" { ... }
}


provider "azurerm" {
  subscription_id = "YOUR_SUBSCRIPTION_ID"

  features {
       resource_group {
       prevent_deletion_if_contains_resources = false
     }
  }
  # Assumes login via Azure CLI (`az login`)
}