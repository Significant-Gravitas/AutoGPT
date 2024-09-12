package database

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/swiftyos/market/config"
	"github.com/swiftyos/market/models"
	"github.com/swiftyos/market/utils"
	"go.uber.org/zap"
)


func NewDB(cfg *config.Config) (*pgxpool.Pool, error) {
	return pgxpool.New(context.Background(), cfg.DatabaseURL)
}

func GetAgents(ctx context.Context, db *pgxpool.Pool, page int, pageSize int, name *string, keywords *string, categories *string) ([]models.Agent, error) {
	logger := zap.L().With(zap.String("function", "GetAgents"))

	logger.Debug("Query parameters",
		zap.Int("page", page),
		zap.Int("pageSize", pageSize),
		zap.String("name", utils.StringOrNil(name)),
		zap.String("keywords", utils.StringOrNil(keywords)),
		zap.String("categories", utils.StringOrNil(categories)))

	query := `
		SELECT * FROM agents
		WHERE submission_status = 'APPROVED'
		AND ($3::text IS NULL OR name ILIKE '%' || $3 || '%')
		AND ($4::text IS NULL OR $4 = ANY(keywords))
		AND ($5::text IS NULL OR $5 = ANY(categories))
		ORDER BY created_at DESC
		LIMIT $1 OFFSET $2
	`

	rows, err := db.Query(ctx, query, pageSize, (page-1)*pageSize, name, keywords, categories)
	if err != nil {
		logger.Error("Error querying agents", zap.Error(err))
		return nil, err
	}
	defer rows.Close()

	var agents []models.Agent
	for rows.Next() {
		var agent models.Agent
		err := rows.Scan(
			&agent.ID,
			&agent.Name,
			&agent.Description,
			&agent.Author,
			&agent.Keywords,
			&agent.Categories,
			&agent.Graph,
		)
		if err != nil {
			logger.Error("Error scanning agent", zap.Error(err))
			return nil, err
		}
		agents = append(agents, agent)
	}
	if err := rows.Err(); err != nil {
		logger.Error("Error iterating over agents", zap.Error(err))
		return nil, err
	}
	logger.Info("Found agents", zap.Int("count", len(agents)))

	return agents, err
}

func SubmitAgent(ctx context.Context, db *pgxpool.Pool, request models.AddAgentRequest, user interface{}) (*models.AgentWithMetadata, error) {
	logger := zap.L().With(zap.String("function", "SubmitAgent"))
	logger.Info("Submitting new agent")

	// Generate a new UUID for the agent
	agentID := uuid.New().String()

	// Create the Agent struct
	agent := models.Agent{
		ID:          agentID,
		Name:        request.Graph.Name,
		Description: request.Graph.Description,
		Author:      request.Author,
		Keywords:    request.Keywords,
		Categories:  request.Categories,
		Graph:       request.Graph,
	}

	// Create the AgentWithMetadata struct
	agentWithMetadata := models.AgentWithMetadata{
		Agent:            agent,
		Version:          1,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
		SubmissionDate:   time.Now(),
		SubmissionStatus: models.SubmissionStatusPending,
	}

	// Start a transaction
	tx, err := db.Begin(ctx)
	if err != nil {
		logger.Error("Failed to begin transaction", zap.Error(err))
		return nil, err
	}
	defer tx.Rollback(ctx)

	// Insert the agent into the database
	_, err = tx.Exec(ctx, `
		INSERT INTO agents (id, name, description, author, keywords, categories, graph, version, created_at, updated_at, submission_date, submission_status)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`, agentWithMetadata.ID, agentWithMetadata.Name, agentWithMetadata.Description, agentWithMetadata.Author,
		agentWithMetadata.Keywords, agentWithMetadata.Categories, agentWithMetadata.Graph,
		agentWithMetadata.Version, agentWithMetadata.CreatedAt, agentWithMetadata.UpdatedAt,
		agentWithMetadata.SubmissionDate, agentWithMetadata.SubmissionStatus)

	if err != nil {
		logger.Error("Failed to insert agent", zap.Error(err))
		return nil, err
	}

	// Commit the transaction
	err = tx.Commit(ctx)
	if err != nil {
		logger.Error("Failed to commit transaction", zap.Error(err))
		return nil, err
	}

	logger.Info("Successfully submitted new agent", zap.String("agentID", agentID))
	return &agentWithMetadata, nil
}

func GetAgentDetails(ctx context.Context, db *pgxpool.Pool, agentID string) (*models.AgentWithMetadata, error) {
	logger := zap.L().With(zap.String("function", "GetAgentDetails"))

	query := `
		SELECT id, name, description, author, keywords, categories, graph, version, created_at, updated_at, submission_date, submission_status
		FROM agents
		WHERE id = $1
	`

	var agent models.AgentWithMetadata
	err := db.QueryRow(ctx, query, agentID).Scan(
		&agent.ID,
		&agent.Name,
		&agent.Description,
		&agent.Author,
		&agent.Keywords,
		&agent.Categories,
		&agent.Graph,
		&agent.Version,
		&agent.CreatedAt,
		&agent.UpdatedAt,
		&agent.SubmissionDate,
		&agent.SubmissionStatus,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			logger.Error("Agent not found", zap.String("agentID", agentID))
			return nil, fmt.Errorf("agent not found")
		}
		logger.Error("Error querying agent details", zap.Error(err))
		return nil, err
	}

	logger.Info("Agent details retrieved", zap.String("agentID", agentID))
	return &agent, nil
}

func IncrementDownloadCount(ctx context.Context, db *pgxpool.Pool, agentID string) error {
	logger := zap.L().With(zap.String("function", "IncrementDownloadCount"))

	query := `
		UPDATE agents
		SET download_count = download_count + 1
		WHERE id = $1
	`

	_, err := db.Exec(ctx, query, agentID)
	if err != nil {
		logger.Error("Failed to increment download count", zap.Error(err), zap.String("agentID", agentID))
		return err
	}

	logger.Info("Download count incremented", zap.String("agentID", agentID))
	return nil
}

func GetAgentFile(ctx context.Context, db *pgxpool.Pool, agentID string) (*models.AgentFile, error) {
	logger := zap.L().With(zap.String("function", "GetAgentFile"))

	query := `
		SELECT id, name, graph
		FROM agents
		WHERE id = $1
	`

	var agentFile models.AgentFile
	err := db.QueryRow(ctx, query, agentID).Scan(
		&agentFile.ID,
		&agentFile.Name,
		&agentFile.Graph,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			logger.Error("Agent not found", zap.String("agentID", agentID))
			return nil, fmt.Errorf("agent not found")
		}
		logger.Error("Error querying agent file", zap.Error(err))
		return nil, err
	}

	logger.Info("Agent file retrieved", zap.String("agentID", agentID))
	return &agentFile, nil
}

