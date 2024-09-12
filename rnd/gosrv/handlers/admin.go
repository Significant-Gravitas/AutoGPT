package handlers

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/swiftyos/market/database"
	"github.com/swiftyos/market/models"
)

func requireAdminUser() gin.HandlerFunc {
	return func(c *gin.Context) {
		user, exists := c.Get("user")
		if !exists {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "User not found in context"})
			return
		}

		userModel, ok := user.(models.User)
		if !ok {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Invalid user model"})
			return
		}

		if userModel.Role != "admin" {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "Admin access required"})
			return
		}

		c.Next()
	}
}

func CreateAgentEntry(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		var request models.AddAgentRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		agent, err := database.CreateAgentEntry(c.Request.Context(), db, models.Agent{
			Name:        request.Graph.Name,
			Description: request.Graph.Description,
			Author:      request.Author,
			Keywords:    request.Keywords,
			Categories:  request.Categories,
			Graph:       request.Graph,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, agent)
	}
}

func SetAgentFeatured(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		agentID := c.Param("agent_id")
		categories := c.QueryArray("categories")
		if len(categories) == 0 {
			categories = []string{"featured"}
		}

		featuredAgent, err := database.SetAgentFeatured(c.Request.Context(), db, agentID, true, categories)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, featuredAgent)
	}
}

func GetAgentFeatured(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		agentID := c.Param("agent_id")

		featuredAgent, err := database.GetAgentFeatured(c.Request.Context(), db, agentID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if featuredAgent == nil {
			c.JSON(http.StatusNotFound, gin.H{"message": "Featured agent not found"})
			return
		}

		c.JSON(http.StatusOK, featuredAgent)
	}
}

func UnsetAgentFeatured(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		agentID := c.Param("agent_id")
		category := c.DefaultQuery("category", "featured")

		featuredAgent, err := database.RemoveFeaturedCategory(c.Request.Context(), db, agentID, category)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if featuredAgent == nil {
			c.JSON(http.StatusNotFound, gin.H{"message": "Featured agent not found"})
			return
		}

		c.JSON(http.StatusOK, featuredAgent)
	}
}

func GetNotFeaturedAgents(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
		pageSize, _ := strconv.Atoi(c.DefaultQuery("page_size", "10"))

		agents, err := database.GetNotFeaturedAgents(c.Request.Context(), db, page, pageSize)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"agents":      agents,
			"total_count": len(agents),
			"page":        page,
			"page_size":   pageSize,
		})
	}
}

func GetAgentSubmissions(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		// TODO: Implement GetAgentSubmissions
		c.JSON(http.StatusNotImplemented, gin.H{"message": "Not Implemented: GetAgentSubmissions"})
	}
}

func ReviewSubmission(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		// TODO: Implement ReviewSubmission
		c.JSON(http.StatusNotImplemented, gin.H{"message": "Not Implemented: ReviewSubmission"})
	}
}

func GetCategories(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		requireAdminUser()(c)
		if c.IsAborted() {
			return
		}

		categories, err := database.GetAllCategories(c.Request.Context(), db)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, categories)
	}
}
