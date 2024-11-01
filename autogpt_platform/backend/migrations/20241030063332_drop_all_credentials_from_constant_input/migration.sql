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

  -- If the JSON contains discord_bot_token
  IF result ? 'discord_bot_token' THEN
    result = result - 'discord_bot_token';
  END IF;

  -- If the JSON contains creds
  IF result ? 'creds' THEN
    result = result - 'creds';
  END IF;

  -- If the JSON contains smtp credentials
  IF result ? 'smtp_username' THEN
    result = result - 'smtp_username';
  END IF;

  IF result ? 'smtp_password' THEN
    result = result - 'smtp_password';
  END IF;

  -- If the JSON contains OAuth credentials
  IF result ? 'client_id' THEN
    result = result - 'client_id';
  END IF;

  IF result ? 'client_secret' THEN
    result = result - 'client_secret';
  END IF;

  -- If the JSON contains username/password
  IF result ? 'username' THEN
    result = result - 'username';
  END IF;

  IF result ? 'password' THEN
    result = result - 'password';
  END IF;

  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Update the table using the function
UPDATE "AgentNode"
SET "constantInput" = clean_sensitive_json("constantInput"::jsonb)::json
WHERE "constantInput"::jsonb ?| array['api_key', 'discord_bot_token', 'creds', 'smtp_username', 'smtp_password', 'client_id', 'client_secret', 'username', 'password'];

-- Drop the function after use
DROP FUNCTION clean_sensitive_json;
