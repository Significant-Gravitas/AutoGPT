package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v4/pgxpool"
	"go.uber.org/zap"

	"github.com/swiftyos/market/database"
	"github.com/swiftyos/market/models"
)

func ListAgents(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement ListAgents
		c.JSON(501, gin.H{"message": "Not Implemented: ListAgents"})
	}
}

func SubmitAgent(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request models.AddAgentRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		user, exists := c.Get("user")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
			return
		}

		agent, err := database.CreateAgentEntry(c.Request.Context(), db, request, user)
		if err != nil {
			logger.Error("Failed to create agent entry", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create agent entry"})
			return
		}

		c.JSON(http.StatusOK, agent)
	}
}
func GetAgentDetails(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetAgentDetails
		c.JSON(501, gin.H{"message": "Not Implemented: GetAgentDetails"})
	}
}

func DownloadAgent(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement DownloadAgent
		c.JSON(501, gin.H{"message": "Not Implemented: DownloadAgent"})
	}
}

func DownloadAgentFile(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement DownloadAgentFile
		c.JSON(501, gin.H{"message": "Not Implemented: DownloadAgentFile"})
	}
}

func TopAgentsByDownloads(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement TopAgentsByDownloads
		c.JSON(501, gin.H{"message": "Not Implemented: TopAgentsByDownloads"})
	}
}

func GetFeaturedAgents(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetFeaturedAgents
		c.JSON(501, gin.H{"message": "Not Implemented: GetFeaturedAgents"})
	}
}
