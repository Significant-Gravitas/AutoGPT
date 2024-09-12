package middleware

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v4"
	"github.com/stretchr/testify/assert"
	"github.com/swiftyos/market/config"
	"github.com/swiftyos/market/models"
)

func TestVerifyUser(t *testing.T) {
	tests := []struct {
		name      string
		payload   jwt.MapClaims
		adminOnly bool
		wantUser  models.User
		wantErr   bool
	}{
		{
			name: "Valid user",
			payload: jwt.MapClaims{
				"sub":   "test-user",
				"email": "test@example.com",
				"role":  "user",
			},
			adminOnly: false,
			wantUser: models.User{
				UserID: "test-user",
				Email:  "test@example.com",
				Role:   "user",
			},
			wantErr: false,
		},
		{
			name: "Valid admin",
			payload: jwt.MapClaims{
				"sub":   "admin-user",
				"email": "admin@example.com",
				"role":  "admin",
			},
			adminOnly: true,
			wantUser: models.User{
				UserID: "admin-user",
				Email:  "admin@example.com",
				Role:   "admin",
			},
			wantErr: false,
		},
		{
			name: "Non-admin accessing admin-only route",
			payload: jwt.MapClaims{
				"sub":   "test-user",
				"email": "test@example.com",
				"role":  "user",
			},
			adminOnly: true,
			wantUser:  models.User{},
			wantErr:   true,
		},
		{
			name:      "Missing sub claim",
			payload:   jwt.MapClaims{},
			adminOnly: false,
			wantUser:  models.User{},
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotUser, err := verifyUser(tt.payload, tt.adminOnly)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantUser, gotUser)
			}
		})
	}
}

func TestParseJWTToken(t *testing.T) {
	secret := "test-secret"

	tests := []struct {
		name        string
		tokenString string
		wantErr     bool
	}{
		{
			name:        "Valid token",
			tokenString: createValidToken(secret),
			wantErr:     false,
		},
		{
			name:        "Invalid token",
			tokenString: "invalid.token.string",
			wantErr:     true,
		},
		{
			name:        "Empty token",
			tokenString: "",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			token, err := parseJWTToken(tt.tokenString, secret)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, token)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, token)
				assert.True(t, token.Valid)
			}
		})
	}
}

func createValidToken(secret string) string {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"sub":   "test-user",
		"email": "test@example.com",
		"role":  "user",
	})
	tokenString, _ := token.SignedString([]byte(secret))
	return tokenString
}

func TestAuth(t *testing.T) {
	cfg := &config.Config{
		JWTSecret:   "test-secret",
		AuthEnabled: true,
	}

	tests := []struct {
		name          string
		authHeader    string
		expectedUser  models.User
		expectedError bool
	}{
		{
			name:       "Valid token",
			authHeader: "Bearer " + createValidToken(cfg.JWTSecret),
			expectedUser: models.User{
				UserID: "test-user",
				Email:  "test@example.com",
				Role:   "user",
			},
			expectedError: false,
		},
		{
			name:          "Invalid token",
			authHeader:    "Bearer invalid.token.string",
			expectedUser:  models.User{},
			expectedError: true,
		},
		{
			name:          "Missing auth header",
			authHeader:    "",
			expectedUser:  models.User{},
			expectedError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock gin.Context
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest("GET", "/", nil)
			c.Request.Header.Set("Authorization", tt.authHeader)

			// Call the Auth middleware
			Auth(cfg)(c)

			// Check the results
			if tt.expectedError {
				assert.True(t, c.IsAborted())
			} else {
				assert.False(t, c.IsAborted())
				user, exists := c.Get("user")
				assert.True(t, exists)
				assert.Equal(t, tt.expectedUser, user.(models.User))
			}
		})
	}
}

func TestAuthDisabled(t *testing.T) {
	cfg := &config.Config{
		JWTSecret:   "test-secret",
		AuthEnabled: false,
	}

	// Create a mock gin.Context
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("GET", "/", nil)

	Auth(cfg)(c)

	assert.False(t, c.IsAborted())
	user, exists := c.Get("user")
	assert.True(t, exists)
	assert.Equal(t, models.User{
		UserID: "3e53486c-cf57-477e-ba2a-cb02dc828e1a",
		Role:   "admin",
	}, user.(models.User))
}
