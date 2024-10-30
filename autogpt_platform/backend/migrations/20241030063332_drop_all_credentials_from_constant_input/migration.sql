-- Function to clean sensitive data from JSON
CREATE OR REPLACE FUNCTION clean_sensitive_json(data jsonb) 
RETURNS jsonb AS $$
DECLARE
  result jsonb := data;
BEGIN
  -- If the JSON contains api_key directly
  IF result ? 'api_key' THEN
    result = result - 'api_key';
  END IF;
  
  -- If the JSON contains credentials
  IF result ? 'credentials' THEN
    result = result - 'credentials';
  END IF;
  
  -- If the JSON contains discord_bot_token
  IF result ? 'discord_bot_token' THEN
    result = result - 'discord_bot_token';
  END IF;
  
  -- If the JSON contains creds
  IF result ? 'creds' THEN
    result = result - 'creds';
  END IF;
  
  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Update the table using the function
UPDATE "AgentNode"
SET "constantInput" = clean_sensitive_json("constantInput"::jsonb)::json
WHERE "constantInput"::jsonb ?| array['api_key', 'credentials', 'discord_bot_token', 'creds'];

-- Drop the function after use
DROP FUNCTION clean_sensitive_json;