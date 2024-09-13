-- Remove sample data from FeaturedAgent table
DELETE FROM "FeaturedAgent" WHERE "agentId" IN ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 'eaa773b1-5efa-485f-b2f0-2e05bae6d297');

-- Remove sample data from InstallTracker table
DELETE FROM "InstallTracker" WHERE "marketplaceAgentId" IN ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 'eaa773b1-5efa-485f-b2f0-2e05bae6d297', 'b47e40a7-ad5f-4b29-9eac-abd5b728f19a', 'a4d3598f-6180-4e6d-96bf-6e15c3de05a9', '9f332ff3-4c74-4f5b-9838-65938a06711f');

-- Remove sample data from AnalyticsTracker table
DELETE FROM "AnalyticsTracker" WHERE "agentId" IN ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 'eaa773b1-5efa-485f-b2f0-2e05bae6d297', 'b47e40a7-ad5f-4b29-9eac-abd5b728f19a', 'a4d3598f-6180-4e6d-96bf-6e15c3de05a9', '9f332ff3-4c74-4f5b-9838-65938a06711f');

-- Remove sample data from Agents table
DELETE FROM "Agents" WHERE "id" IN ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 'eaa773b1-5efa-485f-b2f0-2e05bae6d297', 'b47e40a7-ad5f-4b29-9eac-abd5b728f19a', 'a4d3598f-6180-4e6d-96bf-6e15c3de05a9', '9f332ff3-4c74-4f5b-9838-65938a06711f');