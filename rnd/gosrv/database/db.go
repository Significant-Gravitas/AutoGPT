package database

import (
	"context"
	"fmt"
	"strings"
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
		SELECT * FROM "Agents"
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
		INSERT INTO "Agents" (id, name, description, author, keywords, categories, graph, version, created_at, updated_at, submission_date, submission_status)
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
		FROM "Agents"
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
		UPDATE "Agents"
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
		FROM "Agents"
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

func GetTopAgentsByDownloads(ctx context.Context, db *pgxpool.Pool, page, pageSize int) ([]models.AgentWithDownloads, int, error) {
	logger := zap.L().With(zap.String("function", "GetTopAgentsByDownloads"))

	offset := (page - 1) * pageSize

	query := `
		SELECT a.id, a.name, a.description, a.author, a.keywords, a.categories, a.graph, at.downloads
		FROM "Agents" a
		JOIN "AnalyticsTracker" at ON a.id = at.agent_id
		WHERE a.submission_status = 'APPROVED'
		ORDER BY at.downloads DESC
		LIMIT $1 OFFSET $2
	`

	rows, err := db.Query(ctx, query, pageSize, offset)
	if err != nil {
		logger.Error("Failed to query top agents", zap.Error(err))
		return nil, 0, err
	}
	defer rows.Close()

	var agents []models.AgentWithDownloads
	for rows.Next() {
		var agent models.AgentWithDownloads
		err := rows.Scan(
			&agent.ID,
			&agent.Name,
			&agent.Description,
			&agent.Author,
			&agent.Keywords,
			&agent.Categories,
			&agent.Graph,
			&agent.Downloads,
		)
		if err != nil {
			logger.Error("Failed to scan agent row", zap.Error(err))
			return nil, 0, err
		}
		agents = append(agents, agent)
	}

	var totalCount int
	err = db.QueryRow(ctx, `SELECT COUNT(*) FROM "Agents" WHERE submission_status = 'APPROVED'`).Scan(&totalCount)
	if err != nil {
		logger.Error("Failed to get total count", zap.Error(err))
		return nil, 0, err
	}

	logger.Info("Top agents retrieved", zap.Int("count", len(agents)))
	return agents, totalCount, nil
}

func GetFeaturedAgents(ctx context.Context, db *pgxpool.Pool, category string, page, pageSize int) ([]models.Agent, int, error) {
	logger := zap.L().With(zap.String("function", "GetFeaturedAgents"))

	offset := (page - 1) * pageSize

	query := `
		SELECT a.id, a.name, a.description, a.author, a.keywords, a.categories, a.graph
		FROM "Agents" a
		JOIN "FeaturedAgent" fa ON a.id = fa.agent_id
		WHERE $1 = ANY(fa.featured_categories) AND fa.is_active = true AND a.submission_status = 'APPROVED'
		ORDER BY a.created_at DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := db.Query(ctx, query, category, pageSize, offset)
	if err != nil {
		logger.Error("Failed to query featured agents", zap.Error(err))
		return nil, 0, err
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
			logger.Error("Failed to scan featured agent row", zap.Error(err))
			return nil, 0, err
		}
		agents = append(agents, agent)
	}

	var totalCount int
	err = db.QueryRow(ctx, `SELECT COUNT(*) FROM "FeaturedAgent" fa JOIN "Agents" a ON fa.agent_id = a.id WHERE $1 = ANY(fa.featured_categories) AND fa.is_active = true AND a.submission_status = 'APPROVED'`, category).Scan(&totalCount)
	if err != nil {
		logger.Error("Failed to get total count of featured agents", zap.Error(err))
		return nil, 0, err
	}

	logger.Info("Featured agents retrieved", zap.Int("count", len(agents)))
	return agents, totalCount, nil
}

func Search(ctx context.Context, db *pgxpool.Pool, query string, categories []string, page, pageSize int, sortBy, sortOrder string) ([]models.AgentWithRank, error) {
	logger := zap.L().With(zap.String("function", "Search"))

	offset := (page - 1) * pageSize

	categoryFilter := ""
	if len(categories) > 0 {
		categoryConditions := make([]string, len(categories))
		for i, cat := range categories {
			categoryConditions[i] = fmt.Sprintf("'%s' = ANY(a.categories)", cat)
		}
		categoryFilter = "AND (" + strings.Join(categoryConditions, " OR ") + ")"
	}

	orderByClause := ""
	switch sortBy {
	case "createdAt", "updatedAt":
		orderByClause = fmt.Sprintf(`a."%s" %s, rank DESC`, sortBy, sortOrder)
	case "name":
		orderByClause = fmt.Sprintf(`a.name %s, rank DESC`, sortOrder)
	default:
		orderByClause = `rank DESC, a."createdAt" DESC`
	}

	sqlQuery := fmt.Sprintf(`
		WITH query AS (
			SELECT to_tsquery(string_agg(lexeme || ':*', ' & ' ORDER BY positions)) AS q 
			FROM unnest(to_tsvector($1))
		)
		SELECT 
			a.id, 
			a.created_at, 
			a.updated_at, 
			a.version, 
			a.name, 
			LEFT(a.description, 500) AS description, 
			a.author, 
			a.keywords, 
			a.categories, 
			a.graph,
			a.submission_status,
			a.submission_date,
			ts_rank(CAST(a.search AS tsvector), query.q) AS rank
		FROM "Agents" a, query
		WHERE a.submission_status = 'APPROVED' %s
		ORDER BY %s
		LIMIT $2
		OFFSET $3
	`, categoryFilter, orderByClause)

	rows, err := db.Query(ctx, sqlQuery, query, pageSize, offset)
	if err != nil {
		logger.Error("Failed to execute search query", zap.Error(err))
		return nil, err
	}
	defer rows.Close()

	var agents []models.AgentWithRank
	for rows.Next() {
		var agent models.AgentWithRank
		err := rows.Scan(
			&agent.ID,
			&agent.CreatedAt,
			&agent.UpdatedAt,
			&agent.Version,
			&agent.Name,
			&agent.Description,
			&agent.Author,
			&agent.Keywords,
			&agent.Categories,
			&agent.Graph,
			&agent.SubmissionStatus,
			&agent.SubmissionDate,
			&agent.Rank,
		)
		if err != nil {
			logger.Error("Failed to scan search result row", zap.Error(err))
			return nil, err
		}
		agents = append(agents, agent)
	}

	logger.Info("Search completed", zap.Int("results", len(agents)))
	return agents, nil
}

func CreateAgentInstalledEvent(ctx context.Context, db *pgxpool.Pool, eventData models.InstallTracker) error {
	logger := zap.L().With(zap.String("function", "CreateAgentInstalledEvent"))
	logger.Info("Creating agent installed event")

	query := `
		INSERT INTO install_tracker (marketplace_agent_id, installed_agent_id, installation_location)
		VALUES ($1, $2, $3)
	`

	_, err := db.Exec(ctx, query,
		eventData.MarketplaceAgentID,
		eventData.InstalledAgentID,
		eventData.InstallationLocation,
	)

	if err != nil {
		logger.Error("Failed to create agent installed event", zap.Error(err))
		return fmt.Errorf("failed to create agent installed event: %w", err)
	}

	logger.Info("Agent installed event created successfully")
	return nil
}

// Admin Queries

func CreateAgentEntry(ctx context.Context, db *pgxpool.Pool, agent models.Agent) (models.Agent, error) {
	logger := zap.L().With(zap.String("function", "CreateAgentEntry"))
	logger.Info("Creating agent entry")

	query := `
		INSERT INTO agents (id, name, description, author, keywords, categories, graph)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, name, description, author, keywords, categories, graph
	`
	var createdAgent models.Agent
	err := db.QueryRow(ctx, query,
		agent.ID,
		agent.Name,
		agent.Description,
		agent.Author,
		agent.Keywords,
		agent.Categories,
		agent.Graph,
	).Scan(
		&createdAgent.ID,
		&createdAgent.Name,
		&createdAgent.Description,
		&createdAgent.Author,
		&createdAgent.Keywords,
		&createdAgent.Categories,
		&createdAgent.Graph,
	)

	if err != nil {
		logger.Error("Failed to create agent entry", zap.Error(err))
		return models.Agent{}, err
	}

	logger.Info("Agent entry created successfully", zap.String("agentID", agent.ID))
	return createdAgent, nil
}

func SetAgentFeatured(ctx context.Context, db *pgxpool.Pool, agentID string, isActive bool, featuredCategories []string) (*models.FeaturedAgent, error) {
	logger := zap.L().With(zap.String("function", "SetAgentFeatured"))
	logger.Info("Setting agent featured status", zap.String("agentID", agentID), zap.Bool("isActive", isActive))

	// Check if the agent exists
	var exists bool
	err := db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM "Agents" WHERE id = $1)`, agentID).Scan(&exists)
	if err != nil {
		logger.Error("Failed to check if agent exists", zap.Error(err))
		return nil, fmt.Errorf("failed to check if agent exists: %w", err)
	}
	if !exists {
		return nil, fmt.Errorf("agent with ID %s not found", agentID)
	}

	var query string
	var args []interface{}

	if isActive {
		// Set the agent as featured
		query = `
			INSERT INTO "FeaturedAgent" (agent_id, featured_categories, is_active)
			VALUES ($1, $2, $3)
			ON CONFLICT (agent_id) DO UPDATE
			SET featured_categories = $2, is_active = $3
			RETURNING agent_id, featured_categories, is_active
		`
		args = []interface{}{agentID, featuredCategories, isActive}
	} else {
		// Unset the agent as featured
		query = `
			DELETE FROM "FeaturedAgent"
			WHERE agent_id = $1
			RETURNING agent_id, featured_categories, is_active
		`
		args = []interface{}{agentID}
	}

	var featuredAgent models.FeaturedAgent
	err = db.QueryRow(ctx, query, args...).Scan(
		&featuredAgent.AgentID,
		&featuredAgent.FeaturedCategories,
		&featuredAgent.IsActive,
	)

	if err != nil {
		if err == pgx.ErrNoRows && !isActive {
			logger.Info("Agent was not featured, no action needed", zap.String("agentID", agentID))
			return nil, nil
		}
		logger.Error("Failed to set agent featured status", zap.Error(err))
		return nil, fmt.Errorf("failed to set agent featured status: %w", err)
	}

	if isActive {
		logger.Info("Agent set as featured successfully", zap.String("agentID", agentID))
	} else {
		logger.Info("Agent unset as featured successfully", zap.String("agentID", agentID))
	}
	return &featuredAgent, nil
}

func GetAgentFeatured(ctx context.Context, db *pgxpool.Pool, agentID string) (*models.FeaturedAgent, error) {
	logger := zap.L().With(zap.String("function", "GetAgentFeatured"))
	logger.Info("Getting featured agent", zap.String("agentID", agentID))

	query := `
		SELECT agent_id, featured_categories, is_active
		FROM "FeaturedAgent"
		WHERE agent_id = $1
	`

	var featuredAgent models.FeaturedAgent
	err := db.QueryRow(ctx, query, agentID).Scan(
		&featuredAgent.AgentID,
		&featuredAgent.FeaturedCategories,
		&featuredAgent.IsActive,
	)

	if err == pgx.ErrNoRows {
		return nil, nil
	}

	if err != nil {
		logger.Error("Failed to get featured agent", zap.Error(err))
		return nil, fmt.Errorf("failed to get featured agent: %w", err)
	}

	logger.Info("Featured agent retrieved successfully", zap.String("agentID", agentID))
	return &featuredAgent, nil
}

func RemoveFeaturedCategory(ctx context.Context, db *pgxpool.Pool, agentID string, category string) (*models.FeaturedAgent, error) {
	logger := zap.L().With(zap.String("function", "RemoveFeaturedCategory"))
	logger.Info("Removing featured category", zap.String("agentID", agentID), zap.String("category", category))

	query := `
		UPDATE "FeaturedAgent"
		SET featured_categories = array_remove(featured_categories, $1)
		WHERE agent_id = $2
		RETURNING agent_id, featured_categories, is_active
	`

	var featuredAgent models.FeaturedAgent
	err := db.QueryRow(ctx, query, category, agentID).Scan(
		&featuredAgent.AgentID,
		&featuredAgent.FeaturedCategories,
		&featuredAgent.IsActive,
	)

	if err == pgx.ErrNoRows {
		return nil, nil
	}

	if err != nil {
		logger.Error("Failed to remove featured category", zap.Error(err))
		return nil, fmt.Errorf("failed to remove featured category: %w", err)
	}

	logger.Info("Featured category removed successfully", zap.String("agentID", agentID), zap.String("category", category))
	return &featuredAgent, nil
}

func GetNotFeaturedAgents(ctx context.Context, db *pgxpool.Pool, page, pageSize int) ([]models.Agent, error) {
	logger := zap.L().With(zap.String("function", "GetNotFeaturedAgents"))
	logger.Info("Getting not featured agents", zap.Int("page", page), zap.Int("pageSize", pageSize))

	offset := (page - 1) * pageSize

	query := `
		SELECT a.id, a.name, a.description, a.author, a.keywords, a.categories, a.graph
		FROM "Agents" a
		LEFT JOIN "FeaturedAgent" fa ON a.id = fa.agent_id
		WHERE (fa.agent_id IS NULL OR fa.featured_categories = '{}')
			AND a.submission_status = 'APPROVED'
		ORDER BY a.created_at DESC
		LIMIT $1 OFFSET $2
	`

	rows, err := db.Query(ctx, query, pageSize, offset)
	if err != nil {
		logger.Error("Failed to query not featured agents", zap.Error(err))
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
			logger.Error("Failed to scan not featured agent row", zap.Error(err))
			return nil, err
		}
		agents = append(agents, agent)
	}

	logger.Info("Not featured agents retrieved", zap.Int("count", len(agents)))
	return agents, nil
}

func GetAgentSubmissions(ctx context.Context, db *pgxpool.Pool, page, pageSize int, name, keyword, category *string, sortBy, sortOrder string) ([]models.AgentWithMetadata, int, error) {
	logger := zap.L().With(zap.String("function", "GetAgentSubmissions"))
	logger.Info("Getting agent submissions", zap.Int("page", page), zap.Int("pageSize", pageSize))

	offset := (page - 1) * pageSize

	query := `
		SELECT a.id, a.name, a.description, a.author, a.keywords, a.categories, a.graph, a.created_at, a.updated_at, a.version, a.submission_status, a.submission_review_date, a.submission_review_comments
		FROM "Agents" a
		WHERE a.submission_status = 'PENDING'
	`

	args := []interface{}{}
	argCount := 1

	if name != nil {
		query += fmt.Sprintf(" AND a.name ILIKE $%d", argCount)
		args = append(args, "%"+*name+"%")
		argCount++
	}

	if keyword != nil {
		query += fmt.Sprintf(" AND $%d = ANY(a.keywords)", argCount)
		args = append(args, *keyword)
		argCount++
	}

	if category != nil {
		query += fmt.Sprintf(" AND $%d = ANY(a.categories)", argCount)
		args = append(args, *category)
		argCount++
	}

	// Add sorting
	query += fmt.Sprintf(" ORDER BY a.%s %s", sortBy, sortOrder)

	// Add pagination
	query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argCount, argCount+1)
	args = append(args, pageSize, offset)

	rows, err := db.Query(ctx, query, args...)
	if err != nil {
		logger.Error("Failed to query agent submissions", zap.Error(err))
		return nil, 0, err
	}
	defer rows.Close()

	var agents []models.AgentWithMetadata
	for rows.Next() {
		var agent models.AgentWithMetadata
		err := rows.Scan(
			&agent.ID,
			&agent.Name,
			&agent.Description,
			&agent.Author,
			&agent.Keywords,
			&agent.Categories,
			&agent.Graph,
			&agent.CreatedAt,
			&agent.UpdatedAt,
			&agent.Version,
			&agent.SubmissionStatus,
			&agent.SubmissionReviewDate,
			&agent.SubmissionReviewComments,
		)
		if err != nil {
			logger.Error("Failed to scan agent submission row", zap.Error(err))
			return nil, 0, err
		}
		agents = append(agents, agent)
	}

	// Get total count
	countQuery := `SELECT COUNT(*) FROM "Agents" WHERE submission_status = 'PENDING'`
	var totalCount int
	err = db.QueryRow(ctx, countQuery).Scan(&totalCount)
	if err != nil {
		logger.Error("Failed to get total count of agent submissions", zap.Error(err))
		return nil, 0, err
	}

	logger.Info("Agent submissions retrieved", zap.Int("count", len(agents)))
	return agents, totalCount, nil
}

func ReviewSubmission(ctx context.Context, db *pgxpool.Pool, agentID string, version int, status models.SubmissionStatus, comments *string) (*models.AgentWithMetadata, error) {
	logger := zap.L().With(zap.String("function", "ReviewSubmission"))
	logger.Info("Reviewing agent submission", zap.String("agentID", agentID), zap.Int("version", version))

	query := `
		UPDATE "Agents"
		SET submission_status = $1, 
			submission_review_date = NOW(),
			submission_review_comments = $2
		WHERE id = $3 AND version = $4
		RETURNING id, name, description, author, keywords, categories, graph, created_at, updated_at, version, submission_status, submission_review_date, submission_review_comments
	`

	var agent models.AgentWithMetadata
	err := db.QueryRow(ctx, query, status, comments, agentID, version).Scan(
		&agent.ID,
		&agent.Name,
		&agent.Description,
		&agent.Author,
		&agent.Keywords,
		&agent.Categories,
		&agent.Graph,
		&agent.CreatedAt,
		&agent.UpdatedAt,
		&agent.Version,
		&agent.SubmissionStatus,
		&agent.SubmissionReviewDate,
		&agent.SubmissionReviewComments,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			logger.Error("Agent submission not found", zap.String("agentID", agentID), zap.Int("version", version))
			return nil, fmt.Errorf("agent submission not found")
		}
		logger.Error("Failed to review agent submission", zap.Error(err))
		return nil, err
	}

	logger.Info("Agent submission reviewed successfully", zap.String("agentID", agentID), zap.Int("version", version))
	return &agent, nil
}

func GetAllCategories(ctx context.Context, db *pgxpool.Pool) ([]string, error) {
	logger := zap.L().With(zap.String("function", "GetAllCategories"))
	logger.Info("Getting all categories")

	query := `
		SELECT DISTINCT unnest(categories) AS category
		FROM "Agents"
		ORDER BY category
	`

	rows, err := db.Query(ctx, query)
	if err != nil {
		logger.Error("Failed to query categories", zap.Error(err))
		return nil, err
	}
	defer rows.Close()

	var categories []string
	for rows.Next() {
		var category string
		err := rows.Scan(&category)
		if err != nil {
			logger.Error("Failed to scan category row", zap.Error(err))
			return nil, err
		}
		categories = append(categories, category)
	}

	logger.Info("Categories retrieved", zap.Int("count", len(categories)))
	return categories, nil
}
