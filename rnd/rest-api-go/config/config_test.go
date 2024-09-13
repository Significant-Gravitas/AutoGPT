package config

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadValidConfig(t *testing.T) {
	// Create a temporary config file for testing
	tempFile, err := os.CreateTemp("", "test-config*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write test configuration to the temp file
	testConfig := []byte(`
serveraddress: ":8080"
databaseurl: "postgres://user:pass@localhost:5432/testdb"
authenabled: true
jwtsecret: "test-secret"
jwtalgorithm: "HS256"
`)
	if _, err := tempFile.Write(testConfig); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	// Test the Load function with a specific config file
	config, err := Load(tempFile.Name())
	assert.NoError(t, err)
	assert.NotNil(t, config)

	// Verify the loaded configuration
	assert.Equal(t, ":8080", config.ServerAddress)
	assert.Equal(t, "postgres://user:pass@localhost:5432/testdb", config.DatabaseURL)
	assert.True(t, config.AuthEnabled)
	assert.Equal(t, "test-secret", config.JWTSecret)
	assert.Equal(t, "HS256", config.JWTAlgorithm)
}

func TestLoadDefaultConfigFile(t *testing.T) {
	// Test with default config file (should fail in test environment)
	config, err := Load()
	assert.Error(t, err)
	assert.Nil(t, config)
}

func TestLoadMissingConfigFile(t *testing.T) {
	// Test with missing config file
	config, err := Load("non_existent_config.yaml")
	assert.Error(t, err)
	assert.Nil(t, config)
}

func TestLoadInvalidConfigFormat(t *testing.T) {
	// Create a temporary config file for testing
	tempFile, err := os.CreateTemp("", "test-config*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Test with invalid config format
	invalidConfig := []byte(`
serveraddress: ":8080"
databaseurl: 123  # Invalid type, should be string
`)
	if err := os.WriteFile(tempFile.Name(), invalidConfig, 0644); err != nil {
		t.Fatalf("Failed to write invalid config: %v", err)
	}

	config, err := Load(tempFile.Name())
	assert.Error(t, err)
	assert.Nil(t, config)
}
