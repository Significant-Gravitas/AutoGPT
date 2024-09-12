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
			agents.POST("/submit", middleware.Auth(), handlers.SubmitAgent(db, logger))
		}
	}

	// Start server
	if err := r.Run(cfg.ServerAddress); err != nil {
		logger.Fatal("Failed to start server",  zap.Error(err))
	}
}
