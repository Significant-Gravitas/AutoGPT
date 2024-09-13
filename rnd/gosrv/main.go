package main

import (
	"log"
	"net/http"
	"time"

	"github.com/Depado/ginprom"
	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/zap"
	"github.com/gin-gonic/gin"
	swaggerfiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"github.com/swiftyos/market/config"
	"github.com/swiftyos/market/database"
	docs "github.com/swiftyos/market/docs"
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
	// Set the port
	port := cfg.ServerAddress
	if port == "" {
		port = "8080" // Default port if not specified in config
	}
	r.Run(":" + port)
	p := ginprom.New(
		ginprom.Engine(r),
		ginprom.Subsystem("gin"),
		ginprom.Path("/metrics"),
	)
	r.Use(p.Instrument())
	// Use middleware
	r.Use(ginzap.Ginzap(logger, time.RFC1123, true))
	r.Use(ginzap.RecoveryWithZap(logger, true))
	r.Use(middleware.Gzip())

	// Update CORS configuration
	corsConfig := cors.DefaultConfig()
	if len(cfg.CORSAllowOrigins) > 0 {
		corsConfig.AllowOrigins = cfg.CORSAllowOrigins
	} else {
		corsConfig.AllowOrigins = []string{"*"} // Fallback to allow all origins if not specified
	}
	corsConfig.AllowHeaders = append(corsConfig.AllowHeaders, "Authorization")
	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
	corsConfig.AllowCredentials = true
	r.Use(cors.New(corsConfig))

	// Route welcome
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Welcome to the Marketplace API")
	})
	docs.SwaggerInfo.BasePath = "/api/v1/market/"

	// Setup routes
	// [Error] Request header field Authorization is not allowed by Access-Control-Allow-Headers.
	// [Error] Fetch API cannot load http://localhost:8015/api/v1/market/featured/agents?page=1&page_size=10 due to access control checks.
	// [Error] Failed to load resource: Request header field Authorization is not allowed by Access-Control-Allow-Headers. (agents, line 0)
	api := r.Group("/api/v1/market/")
	{

		agents := api.Group("/agents")
		{
			agents.GET("", handlers.GetAgents(db))
			agents.GET("/:agent_id", handlers.GetAgentDetails(db))
			agents.GET("/:agent_id/download", handlers.DownloadAgent(db))
			agents.GET("/:agent_id/download-file", handlers.DownloadAgentFile(db))
			agents.GET("/top-downloads", handlers.TopAgentsByDownloads(db))
			agents.GET("/featured", handlers.GetFeaturedAgents(db, logger))
			agents.GET("/search", handlers.SearchAgents(db))
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
		}

		admin.GET("/categories", handlers.GetCategories(db))
		// Analytics routes
		api.POST("/agent-installed", handlers.AgentInstalled(db))
	}
	r.GET("/docs/*any", ginSwagger.WrapHandler(swaggerfiles.Handler))

	// Start server
	if err := r.Run(cfg.ServerAddress); err != nil {
		logger.Fatal("Failed to start server", zap.Error(err))
	}
}
