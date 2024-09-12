package handlers

import (
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
)

func AgentInstalled(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement AgentInstalled
		c.JSON(501, gin.H{"message": "Not Implemented: AgentInstalled"})
	}
}
