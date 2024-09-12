package handlers

import (
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
)

func AgentInstalled(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement AgentInstalled
		c.JSON(501, gin.H{"message": "Not Implemented: AgentInstalled"})
	}
}
