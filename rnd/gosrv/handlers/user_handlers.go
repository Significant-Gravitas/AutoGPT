package handlers

import (
	"github.com/gin-gonic/gin"
	"github.com/swiftyos/market/models"
)

func GetUserFromContext(c *gin.Context) (models.User, bool) {
	user, exists := c.Get("user")
	if !exists {
		return models.User{}, false
	}
	return user.(models.User), true
}
