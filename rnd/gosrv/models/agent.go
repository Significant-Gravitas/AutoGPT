package models

import (
	"time"
)

// Agent represents the basic agent information
type Agent struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Author      string   `json:"author"`
	Keywords    []string `json:"keywords"`
	Categories  []string `json:"categories"`
	Graph       Graph    `json:"graph"`
}

// Graph represents the graph structure of an agent
type Graph struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Add other fields as needed
}

// AddAgentRequest represents the request structure for adding a new agent
type AddAgentRequest struct {
	Graph      Graph    `json:"graph"`
	Author     string   `json:"author"`
	Keywords   []string `json:"keywords"`
	Categories []string `json:"categories"`
}

// SubmissionStatus represents the status of an agent submission
type SubmissionStatus string

const (
	SubmissionStatusPending  SubmissionStatus = "PENDING"
	SubmissionStatusApproved SubmissionStatus = "APPROVED"
	SubmissionStatusRejected SubmissionStatus = "REJECTED"
)

// AgentWithMetadata extends Agent with additional metadata
type AgentWithMetadata struct {
	Agent
	Version                  int              `json:"version"`
	CreatedAt                time.Time        `json:"createdAt"`
	UpdatedAt                time.Time        `json:"updatedAt"`
	SubmissionDate           time.Time        `json:"submissionDate"`
	SubmissionReviewDate     *time.Time       `json:"submissionReviewDate,omitempty"`
	SubmissionStatus         SubmissionStatus `json:"submissionStatus"`
	SubmissionReviewComments *string          `json:"submissionReviewComments,omitempty"`
}

// AnalyticsTracker represents analytics data for an agent
type AnalyticsTracker struct {
	ID        string `json:"id"`
	AgentID   string `json:"agentId"`
	Views     int    `json:"views"`
	Downloads int    `json:"downloads"`
}

// InstallationLocation represents the location where an agent is installed
type InstallationLocation string

const (
	InstallationLocationLocal InstallationLocation = "LOCAL"
	InstallationLocationCloud InstallationLocation = "CLOUD"
)

// InstallTracker represents installation data for an agent
type InstallTracker struct {
	ID                   string               `json:"id"`
	MarketplaceAgentID   string               `json:"marketplaceAgentId"`
	InstalledAgentID     string               `json:"installedAgentId"`
	InstallationLocation InstallationLocation `json:"installationLocation"`
	CreatedAt            time.Time            `json:"createdAt"`
}

// FeaturedAgent represents a featured agent in the marketplace
type FeaturedAgent struct {
	ID                 string    `json:"id"`
	AgentID            string    `json:"agentId"`
	IsActive           bool      `json:"isActive"`
	FeaturedCategories []string  `json:"featuredCategories"`
	CreatedAt          time.Time `json:"createdAt"`
	UpdatedAt          time.Time `json:"updatedAt"`
}

type AgentFile struct {
	ID    string      `json:"id"`
	Name  string      `json:"name"`
	Graph interface{} `json:"graph"`
}
