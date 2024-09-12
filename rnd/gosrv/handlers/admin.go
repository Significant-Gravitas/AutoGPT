package handlers

import (
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
)

func CreateAgentEntry(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement CreateAgentEntry
		c.JSON(501, gin.H{"message": "Not Implemented: CreateAgentEntry"})
	}
}

func SetAgentFeatured(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement SetAgentFeatured
		c.JSON(501, gin.H{"message": "Not Implemented: SetAgentFeatured"})
	}
}

func GetAgentFeatured(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetAgentFeatured
		c.JSON(501, gin.H{"message": "Not Implemented: GetAgentFeatured"})
	}
}

func UnsetAgentFeatured(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement UnsetAgentFeatured
		c.JSON(501, gin.H{"message": "Not Implemented: UnsetAgentFeatured"})
	}
}

func GetNotFeaturedAgents(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetNotFeaturedAgents
		c.JSON(501, gin.H{"message": "Not Implemented: GetNotFeaturedAgents"})
	}
}

func GetAgentSubmissions(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetAgentSubmissions
		c.JSON(501, gin.H{"message": "Not Implemented: GetAgentSubmissions"})
	}
}

func ReviewSubmission(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement ReviewSubmission
		c.JSON(501, gin.H{"message": "Not Implemented: ReviewSubmission"})
	}
}

func GetCategories(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement GetCategories
		c.JSON(501, gin.H{"message": "Not Implemented: GetCategories"})
	}
}
