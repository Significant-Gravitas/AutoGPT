package middleware

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v4"
	"github.com/swiftyos/market/config"
	"github.com/swiftyos/market/models"
)

func Auth(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !cfg.AuthEnabled {
			// This handles the case when authentication is disabled
			defaultUser := models.User{
				UserID: "3e53486c-cf57-477e-ba2a-cb02dc828e1a",
				Role:   "admin",
			}
			c.Set("user", defaultUser)
			c.Next()
			return
		}

		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Authorization header is missing"})
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		token, err := parseJWTToken(tokenString, cfg.JWTSecret)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok || !token.Valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			return
		}

		user, err := verifyUser(claims, false) // Pass 'true' for admin-only routes
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
			return
		}

		c.Set("user", user)
		c.Next()
	}
}

func verifyUser(payload jwt.MapClaims, adminOnly bool) (models.User, error) {
	user, err := models.NewUserFromPayload(payload)
	if err != nil {
		return models.User{}, err
	}

	if adminOnly && user.Role != "admin" {
		return models.User{}, errors.New("Admin access required")
	}

	return user, nil
}

func parseJWTToken(tokenString string, secret string) (*jwt.Token, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return []byte(secret), nil
	})

	if err != nil {
		return nil, err
	}

	return token, nil
}
