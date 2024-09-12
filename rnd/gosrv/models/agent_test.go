package models

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestAgentJSON(t *testing.T) {
	jsonStr := `{
		"id": "test-id",
		"name": "Test Agent",
		"description": "A test agent",
		"author": "Test Author",
		"keywords": ["test", "agent"],
		"categories": ["testing"],
		"graph": {
			"name": "Test Graph",
			"description": "A test graph"
		}
	}`

	var agent Agent
	err := json.Unmarshal([]byte(jsonStr), &agent)
	assert.NoError(t, err)

	assert.Equal(t, "test-id", agent.ID)
	assert.Equal(t, "Test Agent", agent.Name)
	assert.Equal(t, "A test agent", agent.Description)
	assert.Equal(t, "Test Author", agent.Author)
	assert.Equal(t, []string{"test", "agent"}, agent.Keywords)
	assert.Equal(t, []string{"testing"}, agent.Categories)
	assert.Equal(t, "Test Graph", agent.Graph.Name)
	assert.Equal(t, "A test graph", agent.Graph.Description)
}

func TestGraphJSON(t *testing.T) {
	jsonStr := `{
		"name": "Test Graph",
		"description": "A test graph"
	}`

	var graph Graph
	err := json.Unmarshal([]byte(jsonStr), &graph)
	assert.NoError(t, err)

	assert.Equal(t, "Test Graph", graph.Name)
	assert.Equal(t, "A test graph", graph.Description)
}

func TestAddAgentRequestJSON(t *testing.T) {
	jsonStr := `{
		"graph": {
			"name": "Test Graph",
			"description": "A test graph"
		},
		"author": "Test Author",
		"keywords": ["test", "agent"],
		"categories": ["testing"]
	}`

	var request AddAgentRequest
	err := json.Unmarshal([]byte(jsonStr), &request)
	assert.NoError(t, err)

	assert.Equal(t, "Test Graph", request.Graph.Name)
	assert.Equal(t, "A test graph", request.Graph.Description)
	assert.Equal(t, "Test Author", request.Author)
	assert.Equal(t, []string{"test", "agent"}, request.Keywords)
	assert.Equal(t, []string{"testing"}, request.Categories)
}

func TestAgentWithMetadataJSON(t *testing.T) {
	now := time.Now().UTC().Round(time.Second)
	jsonStr := `{
		"id": "test-id",
		"name": "Test Agent",
		"description": "A test agent",
		"author": "Test Author",
		"keywords": ["test", "agent"],
		"categories": ["testing"],
		"graph": {
			"name": "Test Graph",
			"description": "A test graph"
		},
		"version": 1,
		"createdAt": "` + now.Format(time.RFC3339) + `",
		"updatedAt": "` + now.Format(time.RFC3339) + `",
		"submissionDate": "` + now.Format(time.RFC3339) + `",
		"submissionStatus": "PENDING"
	}`

	var agent AgentWithMetadata
	err := json.Unmarshal([]byte(jsonStr), &agent)
	assert.NoError(t, err)

	assert.Equal(t, "test-id", agent.ID)
	assert.Equal(t, "Test Agent", agent.Name)
	assert.Equal(t, "A test agent", agent.Description)
	assert.Equal(t, "Test Author", agent.Author)
	assert.Equal(t, []string{"test", "agent"}, agent.Keywords)
	assert.Equal(t, []string{"testing"}, agent.Categories)
	assert.Equal(t, "Test Graph", agent.Graph.Name)
	assert.Equal(t, "A test graph", agent.Graph.Description)
	assert.Equal(t, 1, agent.Version)
	assert.Equal(t, now, agent.CreatedAt)
	assert.Equal(t, now, agent.UpdatedAt)
	assert.Equal(t, now, agent.SubmissionDate)
	assert.Equal(t, SubmissionStatusPending, agent.SubmissionStatus)
	assert.Nil(t, agent.SubmissionReviewDate)
	assert.Nil(t, agent.SubmissionReviewComments)
}

func TestAnalyticsTrackerJSON(t *testing.T) {
	jsonStr := `{
		"id": "tracker-id",
		"agentId": "agent-id",
		"views": 100,
		"downloads": 50
	}`

	var tracker AnalyticsTracker
	err := json.Unmarshal([]byte(jsonStr), &tracker)
	assert.NoError(t, err)

	assert.Equal(t, "tracker-id", tracker.ID)
	assert.Equal(t, "agent-id", tracker.AgentID)
	assert.Equal(t, 100, tracker.Views)
	assert.Equal(t, 50, tracker.Downloads)
}

func TestInstallTrackerJSON(t *testing.T) {
	now := time.Now().UTC().Round(time.Second)
	jsonStr := `{
		"id": "install-id",
		"marketplaceAgentId": "marketplace-agent-id",
		"installedAgentId": "installed-agent-id",
		"installationLocation": "LOCAL",
		"createdAt": "` + now.Format(time.RFC3339) + `"
	}`

	var tracker InstallTracker
	err := json.Unmarshal([]byte(jsonStr), &tracker)
	assert.NoError(t, err)

	assert.Equal(t, "install-id", tracker.ID)
	assert.Equal(t, "marketplace-agent-id", tracker.MarketplaceAgentID)
	assert.Equal(t, "installed-agent-id", tracker.InstalledAgentID)
	assert.Equal(t, InstallationLocationLocal, tracker.InstallationLocation)
	assert.Equal(t, now, tracker.CreatedAt)
}

func TestFeaturedAgentJSON(t *testing.T) {
	now := time.Now().UTC().Round(time.Second)
	jsonStr := `{
		"id": "featured-id",
		"agentId": "agent-id",
		"isActive": true,
		"featuredCategories": ["category1", "category2"],
		"createdAt": "` + now.Format(time.RFC3339) + `",
		"updatedAt": "` + now.Format(time.RFC3339) + `"
	}`

	var featured FeaturedAgent
	err := json.Unmarshal([]byte(jsonStr), &featured)
	assert.NoError(t, err)

	assert.Equal(t, "featured-id", featured.ID)
	assert.Equal(t, "agent-id", featured.AgentID)
	assert.True(t, featured.IsActive)
	assert.Equal(t, []string{"category1", "category2"}, featured.FeaturedCategories)
	assert.Equal(t, now, featured.CreatedAt)
	assert.Equal(t, now, featured.UpdatedAt)
}
