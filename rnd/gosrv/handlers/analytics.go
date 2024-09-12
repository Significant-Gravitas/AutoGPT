package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
	"github.com/swiftyos/market/database"

	"github.com/swiftyos/market/models"
)

func AgentInstalled(db *pgxpool.Pool) gin.HandlerFunc {
	return func(c *gin.Context) {
		logger := zap.L().With(zap.String("function", "AgentInstalled"))
		var eventData models.InstallTracker
		if err := c.ShouldBindJSON(&eventData); err != nil {
			logger.Error("Failed to bind JSON", zap.Error(err))
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
			return
		}

		err := database.CreateAgentInstalledEvent(c.Request.Context(), db, models.InstallTracker{
			MarketplaceAgentID:    eventData.MarketplaceAgentID,
			InstalledAgentID:      eventData.InstalledAgentID,
			InstallationLocation:  eventData.InstallationLocation,
		})
		if err != nil {
			logger.Error("Failed to process agent installed event", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process agent installed event"})
			return
		}
		logger.Info("Agent installed event processed successfully",
			zap.String("marketplaceAgentID", eventData.MarketplaceAgentID),
			zap.String("installedAgentID", eventData.InstalledAgentID),
			zap.String("installationLocation", string(eventData.InstallationLocation)))

		c.JSON(http.StatusOK, gin.H{"message": "Agent installed event processed successfully"})
	}
}
