#!/bin/bash

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Validate GitHub username
if [[ ! $GITHUB_USERNAME =~ ^[a-zA-Z0-9](-?[a-zA-Z0-9])*$ ]]; then
    echo "Invalid GitHub username. It should contain only alphanumeric characters and hyphens, and cannot start or end with a hyphen."
    exit 1
fi

# Prompt for project name
read -p "Enter your project name (e.g., myproject): " PROJECT_NAME

# Validate project name
if [[ ! $PROJECT_NAME =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "Invalid project name. It should start with a lowercase letter and contain only lowercase letters, numbers, and underscores."
    exit 1
fi

# Create project directory
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create directory structure
mkdir -p config middleware models handlers database utils

# Create main.go
cat << EOF > main.go
package main

import (
	"log"

	"github.com/gin-gonic/gin"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/config"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/database"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/handlers"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/middleware"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/utils"
)

func main() {
	// Initialize configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize logger
	logger := utils.NewLogger(cfg)

	// Initialize database connection
	db, err := database.NewDB(cfg)
	if err != nil {
		logger.Fatal("Failed to connect to database", "error", err)
	}

	// Initialize Gin router
	r := gin.New()

	// Use middleware
	r.Use(gin.Recovery())
	r.Use(middleware.Logger(logger))
	r.Use(middleware.Gzip())

	// Setup routes
	api := r.Group("/api")
	{
		agents := api.Group("/agents")
		{
			agents.POST("/submit", middleware.Auth(), handlers.SubmitAgent(db, logger))
		}
	}

	// Start server
	if err := r.Run(cfg.ServerAddress); err != nil {
		logger.Fatal("Failed to start server", "error", err)
	}
}
EOF

# Create config/config.go
cat << EOF > config/config.go
package config

import (
	"github.com/spf13/viper"
)

type Config struct {
	ServerAddress string
	DatabaseURL   string
	JWTSecret     string
	JWTAlgorithm  string
}

func Load() (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")

	if err := viper.ReadInConfig(); err != nil {
		return nil, err
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, err
	}

	return &config, nil
}
EOF

# Create middleware/auth.go
cat << EOF > middleware/auth.go
package middleware

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v4"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/config"
)

func Auth() gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Authorization header is missing"})
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		token, err := parseJWTToken(tokenString)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok || !token.Valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			return
		}

		c.Set("user", claims)
		c.Next()
	}
}

func parseJWTToken(tokenString string) (*jwt.Token, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return []byte(cfg.JWTSecret), nil
	})

	if err != nil {
		return nil, err
	}

	return token, nil
}
EOF

# Create middleware/gzip.go
cat << EOF > middleware/gzip.go
package middleware

import (
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
)

func Gzip() gin.HandlerFunc {
	return gzip.Gzip(gzip.DefaultCompression)
}
EOF

# Create models/agent.go
cat << EOF > models/agent.go
package models

type Agent struct {
	ID          string   \`json:"id"\`
	Name        string   \`json:"name"\`
	Description string   \`json:"description"\`
	Author      string   \`json:"author"\`
	Keywords    []string \`json:"keywords"\`
	Categories  []string \`json:"categories"\`
	Graph       Graph    \`json:"graph"\`
}

type Graph struct {
	Name        string \`json:"name"\`
	Description string \`json:"description"\`
	// Add other fields as needed
}

type AddAgentRequest struct {
	Graph      Graph    \`json:"graph"\`
	Author     string   \`json:"author"\`
	Keywords   []string \`json:"keywords"\`
	Categories []string \`json:"categories"\`
}
EOF

# Create handlers/agent_handlers.go
cat << EOF > handlers/agent_handlers.go
package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v4/pgxpool"
	"go.uber.org/zap"

	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/database"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/models"
)

func SubmitAgent(db *pgxpool.Pool, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request models.AddAgentRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		user, exists := c.Get("user")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
			return
		}

		agent, err := database.CreateAgentEntry(c.Request.Context(), db, request, user)
		if err != nil {
			logger.Error("Failed to create agent entry", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create agent entry"})
			return
		}

		c.JSON(http.StatusOK, agent)
	}
}
EOF

# Create database/db.go
cat << EOF > database/db.go
package database

import (
	"context"

	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/config"
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/models"
)

func NewDB(cfg *config.Config) (*pgxpool.Pool, error) {
	return pgxpool.Connect(context.Background(), cfg.DatabaseURL)
}

func CreateAgentEntry(ctx context.Context, db *pgxpool.Pool, request models.AddAgentRequest, user interface{}) (*models.Agent, error) {
	// Implement the database logic to create an agent entry
	// Use the provided db connection pool to execute queries
	// Return the created agent or an error
	return nil, nil // Replace with actual implementation
}
EOF

# Create utils/logger.go
cat << EOF > utils/logger.go
package utils

import (
	"github.com/$GITHUB_USERNAME/$PROJECT_NAME/config"
	"go.uber.org/zap"
)

func NewLogger(cfg *config.Config) *zap.Logger {
	logger, _ := zap.NewProduction()
	return logger
}
EOF

# Create go.mod file
cat << EOF > go.mod
module github.com/$GITHUB_USERNAME/$PROJECT_NAME

go 1.17

require (
	github.com/gin-gonic/gin v1.7.7
	github.com/golang-jwt/jwt/v4 v4.4.1
	github.com/jackc/pgx/v4 v4.16.1
	github.com/spf13/viper v1.11.0
	go.uber.org/zap v1.21.0
)
EOF

# Create config.yaml file
cat << EOF > config.yaml
ServerAddress: ":8080"
DatabaseURL: "postgres://username:password@localhost:5432/database?sslmode=disable"
JWTSecret: "your-secret-key"
JWTAlgorithm: "HS256"
EOF

echo "Project structure created successfully!"
echo "GitHub Username: $GITHUB_USERNAME"
echo "Project name: $PROJECT_NAME"
echo "To get started:"
echo "1. cd $PROJECT_NAME"
echo "2. go mod tidy"
echo "3. Update config.yaml with your specific configuration"
echo "4. Implement database operations in database/db.go"
echo "5. Add additional routes, handlers, or models as needed"
EOF
