package main

import (
	"log"
	"time"

	"github.com/gin-contrib/zap"
	"github.com/gin-gonic/gin"
	"github.com/swiftyos/market/config"
	"github.com/swiftyos/market/database"
	"github.com/swiftyos/market/handlers"
	"github.com/swiftyos/market/middleware"
	"github.com/swiftyos/market/utils"
	"go.uber.org/zap"
)

func main() {
	// Initialize configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize logger
	logger := utils.NewLogger(cfg)

	// Initialize database connection
	db, err := database.NewDB(cfg)
	if err != nil {
		logger.Fatal("Failed to connect to database", zap.Error(err))
	}
	// Initialize Gin router
	r := gin.New()

	// Use middleware
	r.Use(ginzap.Ginzap(logger, time.RFC1123, true))
	r.Use(ginzap.RecoveryWithZap(logger, true))
	r.Use(middleware.Gzip())

	// Setup routes
	api := r.Group("/api")
	{
		agents := api.Group("/agents")
		{
			agents.GET("", handlers.ListAgents(db))
			agents.GET("/:agent_id", handlers.GetAgentDetails(db))
			agents.GET("/:agent_id/download", handlers.DownloadAgent(db))
			agents.GET("/:agent_id/download-file", handlers.DownloadAgentFile(db))
			agents.GET("/top-downloads", handlers.TopAgentsByDownloads(db))
			agents.GET("/featured", handlers.GetFeaturedAgents(db))
			agents.GET("/search", handlers.Search(db))
			agents.POST("/submit", middleware.Auth(cfg), handlers.SubmitAgent(db))
		}

		// Admin routes
		admin := api.Group("/admin")
		{
			admin.POST("/agent", middleware.Auth(cfg), handlers.CreateAgentEntry(db))
			admin.POST("/agent/featured/:agent_id", middleware.Auth(cfg), handlers.SetAgentFeatured(db))
			admin.GET("/agent/featured/:agent_id", middleware.Auth(cfg), handlers.GetAgentFeatured(db))
			admin.DELETE("/agent/featured/:agent_id", middleware.Auth(cfg), handlers.UnsetAgentFeatured(db))
			admin.GET("/agent/not-featured", middleware.Auth(cfg), handlers.GetNotFeaturedAgents(db))
			admin.GET("/agent/submissions", middleware.Auth(cfg), handlers.GetAgentSubmissions(db))
			admin.POST("/agent/submissions", middleware.Auth(cfg), handlers.ReviewSubmission(db))
			admin.GET("/categories", handlers.GetCategories(db))
		}

		// Analytics routes
		api.POST("/agent-installed", handlers.AgentInstalled(db))
	}

	// Start server
	if err := r.Run(cfg.ServerAddress); err != nil {
		logger.Fatal("Failed to start server", zap.Error(err))
	}
}
