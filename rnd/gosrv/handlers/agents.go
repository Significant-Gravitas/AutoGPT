package handlers

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"github.com/swiftyos/market/database"
	"github.com/swiftyos/market/models"
	"github.com/swiftyos/market/utils"
)

func ListAgents(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "ListAgents"))
		// Get pagination parameters from context
		page := getPageFromContext(c.Request.Context())
		pageSize := getPageSizeFromContext(c.Request.Context())

		// Get filter parameters from context
		name := getNameFromContext(c.Request.Context())
		keywords := getKeywordsFromContext(c.Request.Context())
		categories := getCategoriesFromContext(c.Request.Context())

		logger.Debug("Request parameters",
			zap.Int("page", page),
			zap.Int("pageSize", pageSize),
			zap.String("name", utils.StringOrNil(name)),
			zap.String("keywords", utils.StringOrNil(keywords)),
			zap.String("categories", utils.StringOrNil(categories)))

		agents, err := database.GetAgents(c.Request.Context(), db, page, pageSize, name, keywords, categories)
		if err != nil {
			logger.Error("Failed to fetch agents", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch agents"})
			return
		}

		c.JSON(http.StatusOK, agents)
	}
}

func SubmitAgent(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "SubmitAgent"))
		var request models.AddAgentRequest
		logger.Debug("Add Agent Request body", zap.Any("request", request))
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		user, exists := c.Get("user")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
			return
		}

		agent, err := database.SubmitAgent(c.Request.Context(), db, request, user)
		if err != nil {
			logger.Error("Failed to submit agent", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to submit agent"})
			return
		}

		c.JSON(http.StatusOK, agent)
	}
}

func GetAgentDetails(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "GetAgentDetails"))

		agentID := c.Param("id")
		logger.Debug("Agent ID", zap.String("agentID", agentID))
		
		if agentID == "" {
			logger.Error("Agent ID is required")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Agent ID is required"})
			return
		}

		agent, err := database.GetAgentDetails(c.Request.Context(), db, agentID)
		if err != nil {
			if err.Error() == "agent not found" {
				logger.Error("Agent not found", zap.String("agentID", agentID))
				c.JSON(http.StatusNotFound, gin.H{"error": "Agent not found"})
				return
			}
			logger.Error("Failed to fetch agent details", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch agent details"})
			return
		}

		c.JSON(http.StatusOK, agent)
	}
}

func DownloadAgent(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "DownloadAgent"))

		agentID := c.Param("id")
		if agentID == "" {
			logger.Error("Agent ID is required")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Agent ID is required"})
			return
		}

		agent, err := database.GetAgentDetails(c.Request.Context(), db, agentID)
		if err != nil {
			if err.Error() == "agent not found" {
				logger.Error("Agent not found", zap.String("agentID", agentID))
				c.JSON(http.StatusNotFound, gin.H{"error": "Agent not found"})
				return
			}
			logger.Error("Failed to fetch agent details", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch agent details"})
			return
		}

		err = database.IncrementDownloadCount(c.Request.Context(), db, agentID)
		if err != nil {
			logger.Error("Failed to increment download count", zap.Error(err))
			// Continue with the download even if the count update fails
		}

		c.JSON(http.StatusOK, agent)
	}
}

func DownloadAgentFile(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "DownloadAgentFile"))

		agentID := c.Param("id")
		if agentID == "" {
			logger.Error("Agent ID is required")
			c.JSON(http.StatusBadRequest, gin.H{"error": "Agent ID is required"})
			return
		}

		agentFile, err := database.GetAgentFile(c.Request.Context(), db, agentID)
		if err != nil {
			if err.Error() == "agent not found" {
				logger.Error("Agent not found", zap.String("agentID", agentID))
				c.JSON(http.StatusNotFound, gin.H{"error": "Agent not found"})
				return
			}
			logger.Error("Failed to fetch agent file", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch agent file"})
			return
		}

		err = database.IncrementDownloadCount(c.Request.Context(), db, agentID)
		if err != nil {
			logger.Error("Failed to increment download count", zap.Error(err))
			// Continue with the download even if the count update fails
		}

		fileName := fmt.Sprintf("agent_%s.json", agentID)
		c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=%s", fileName))
		c.JSON(http.StatusOK, agentFile.Graph)
	}
}

func TopAgentsByDownloads(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		
		// TODO: Implement the database function to get top agents by downloads
		// agents, err := database.GetTopAgentsByDownloads(c.Request.Context(), db, page, pageSize)
		// if err != nil {
		// 	logger.Error("Failed to fetch top agents", zap.Error(err))
		// 	c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch top agents"})
		// 	return
		// }

		// c.JSON(http.StatusOK, agents)

		// For now, return a placeholder response
		c.JSON(http.StatusOK, gin.H{"message": "Top agents by downloads will be implemented soon"})
	}
}

func GetFeaturedAgents(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetFeaturedAgents
		c.JSON(501, gin.H{"message": "Not Implemented: GetFeaturedAgents"})
	}
}
