-- Add priceTier column to LlmModel table
-- This extends model metadata for the LLM Picker UI
-- priceTier: 1=cheapest, 2=medium, 3=expensive

ALTER TABLE "LlmModel" ADD COLUMN IF NOT EXISTS "priceTier" INTEGER NOT NULL DEFAULT 1;
