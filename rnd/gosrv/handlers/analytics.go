package handlers

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	"github.com/jackc/pgx/v4/pgxpool"
)

func AgentInstalled(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement AgentInstalled
		c.JSON(501, gin.H{"message": "Not Implemented: AgentInstalled"})
	}
}