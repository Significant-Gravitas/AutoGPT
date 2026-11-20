-- The Local PC companion's own OAuth scope: it lets a computer connect as the
-- user's Local PC executor and grants nothing else. The companion used to ask
-- for USE_TOOLS, which also covers External API tools. API keys never carry it.
-- Adding an enum value takes no lock on the tables that use the type.
ALTER TYPE "APIKeyPermission" ADD VALUE 'CONNECT_LOCAL_EXECUTOR';
