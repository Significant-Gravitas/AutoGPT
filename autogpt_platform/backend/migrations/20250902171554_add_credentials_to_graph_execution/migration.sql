-- Add 'credentialInputs', 'inputs', and 'nodesInputMasks' columns to the AgentGraphExecution table
ALTER TABLE "AgentGraphExecution"
  ADD COLUMN "credentialInputs" JSONB,
  ADD COLUMN "inputs"           JSONB,
  ADD COLUMN "nodesInputMasks"  JSONB;
