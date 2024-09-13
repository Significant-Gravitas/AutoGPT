-- Sample data for Agents table (10 agents)

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph", "submissionStatus")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', 'AI Recruiter', 'An AI-powered tool that assists HR teams with talent acquisition, screening, and shortlisting.', 'Author1', ARRAY['recruitment', 'HR'], ARRAY['human resources', 'talent management'], '{"key": "value"}', 'APPROVED');

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph", "submissionStatus")
VALUES ('3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 'Customer Service Bot', 'A chatbot that provides 24/7 support and assistance to customers, handling common inquiries and issues.', 'Author2', ARRAY['customer service', 'chatbot'], ARRAY['customer experience', 'artificial intelligence'], '{"key": "value"}', 'APPROVED');

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph", "submissionStatus")
VALUES ('eaa773b1-5efa-485f-b2f0-2e05bae6d297', 'Financial Advisor', 'An AI-powered financial advisor that offers personalized investment recommendations and portfolio management.', 'Author3', ARRAY['finance', 'investment'], ARRAY['wealth management', 'artificial intelligence'], '{"key": "value"}', 'APPROVED');

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph", "submissionStatus")
VALUES ('b47e40a7-ad5f-4b29-9eac-abd5b728f19a', 'AI Content Writer', 'An AI-powered tool that generates high-quality content for websites, blogs, and marketing materials.', 'Author4', ARRAY['content writing', 'AI'], ARRAY['marketing', 'artificial intelligence'], '{"key": "value"}', 'APPROVED');

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph", "submissionStatus")
VALUES ('a4d3598f-6180-4e6d-96bf-6e15c3de05a9', 'AI Image Generator', 'An AI-powered tool that creates realistic images based on text prompts.', 'Author5', ARRAY['image generation', 'AI'], ARRAY['marketing', 'artificial intelligence'], '{"key": "value"}', 'APPROVED');

INSERT INTO "Agents" ("id", "name", "description", "author", "keywords", "categories", "graph")
VALUES ('9f332ff3-4c74-4f5b-9838-65938a06711f', 'AI Video Editor', 'An AI-powered tool that edits and enhances videos with advanced AI algorithms.', 'Author6', ARRAY['video editing', 'AI'], ARRAY['marketing', 'artificial intelligence'], '{"key": "value"}');

-- Sample data for AnalyticsTracker table (10 agents)
INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', 200, 80);

INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', 150, 60);

INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('eaa773b1-5efa-485f-b2f0-2e05bae6d297', 100, 40);

INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('b47e40a7-ad5f-4b29-9eac-abd5b728f19a', 120, 50);

INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('a4d3598f-6180-4e6d-96bf-6e15c3de05a9', 130, 55);

INSERT INTO "AnalyticsTracker" ("agentId", "views", "downloads")
VALUES ('9f332ff3-4c74-4f5b-9838-65938a06711f', 140, 60);


-- Sample data for InstallTracker table (10 agents)
INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac11', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac12', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac13', 'LOCAL');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac14', 'LOCAL');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac15', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac16', 'LOCAL');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', '244f809e-1eee-4a36-a49b-ac2db008ac17', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', '244f809e-1eee-4a36-a49b-ac2db008ac18', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('eaa773b1-5efa-485f-b2f0-2e05bae6d297', '244f809e-1eee-4a36-a49b-ac2db008ac19', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('b47e40a7-ad5f-4b29-9eac-abd5b728f19a', '244f809e-1eee-4a36-a49b-ac2db008ac20', 'LOCAL');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('a4d3598f-6180-4e6d-96bf-6e15c3de05a9', '244f809e-1eee-4a36-a49b-ac2db008ac22', 'CLOUD');

INSERT INTO "InstallTracker" ("marketplaceAgentId", "installedAgentId", "installationLocation")
VALUES ('9f332ff3-4c74-4f5b-9838-65938a06711f', '244f809e-1eee-4a36-a49b-ac2db008ac21', 'CLOUD');

-- Sample data for FeaturedAgent table (3 featured agents)
INSERT INTO "FeaturedAgent" ("agentId", "isActive", "featuredCategories")
VALUES ('b609e5fd-c992-4be9-b68f-afc1980f93c0', true, ARRAY['human resources', 'talent management']);

INSERT INTO "FeaturedAgent" ("agentId", "isActive", "featuredCategories")
VALUES ('3b6d8f75-99d3-41e3-b484-4b2c5f835f5b', true, ARRAY['customer experience', 'artificial intelligence']);

INSERT INTO "FeaturedAgent" ("agentId", "isActive", "featuredCategories")
VALUES ('eaa773b1-5efa-485f-b2f0-2e05bae6d297', true, ARRAY['wealth management', 'artificial intelligence']);
