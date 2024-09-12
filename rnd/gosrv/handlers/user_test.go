package handlers

import (
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/swiftyos/market/models"
)

func TestGetUserFromContext(t *testing.T) {
	t.Run("User exists in context", func(t *testing.T) {
		// Create a new gin context
		c, _ := gin.CreateTestContext(nil)

		// Create a test user
		testUser := models.User{
			UserID: "123",
			Role:   "admin",
			Email:  "test@example.com",
		}

		// Set the user in the context
		c.Set("user", testUser)

		// Call the function
		user, exists := GetUserFromContext(c)

		// Assert the results
		assert.True(t, exists)
		assert.Equal(t, testUser, user)
	})

	t.Run("User does not exist in context", func(t *testing.T) {
		// Create a new gin context
		c, _ := gin.CreateTestContext(nil)

		// Call the function
		user, exists := GetUserFromContext(c)

		// Assert the results
		assert.False(t, exists)
		assert.Equal(t, models.User{}, user)
	})
}
