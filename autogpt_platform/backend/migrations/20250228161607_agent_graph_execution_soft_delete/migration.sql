-- Add isDeleted column to AgentGraphExecution
ALTER TABLE "AgentGraphExecution"
ADD COLUMN  "isDeleted"
  BOOLEAN
  NOT NULL
  DEFAULT false;
