package config

import (
	"fmt"

	"github.com/spf13/viper"
	"go.uber.org/zap"
)

type Config struct {
	ServerAddress    string   `mapstructure:"serveraddress"`
	DatabaseURL      string   `mapstructure:"databaseurl"`
	AuthEnabled      bool     `mapstructure:"authenabled"`
	JWTSecret        string   `mapstructure:"jwtsecret"`
	JWTAlgorithm     string   `mapstructure:"jwtalgorithm"`
	CORSAllowOrigins []string `mapstructure:"corsalloworigins"`
}

func Load(configFile ...string) (*Config, error) {
	logger := zap.L().With(zap.String("function", "Load"))

	if len(configFile) > 0 {
		viper.SetConfigFile(configFile[0])
	} else {
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
		viper.AddConfigPath(".")
	}

	viper.SetEnvPrefix("AGPT")
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err != nil {
		logger.Error("Failed to read config file", zap.Error(err))
		return nil, err
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		logger.Error("Failed to unmarshal config", zap.Error(err))
		return nil, err
	}

	// Validate required fields
	if config.ServerAddress == "" {
		logger.Error("ServerAddress is required")
		return nil, fmt.Errorf("ServerAddress is required")
	}
	if config.DatabaseURL == "" {
		logger.Error("DatabaseURL is required")
		return nil, fmt.Errorf("DatabaseURL is required")
	}
	if config.JWTSecret == "" {
		logger.Error("JWTSecret is required")
		return nil, fmt.Errorf("JWTSecret is required")
	}
	if config.JWTAlgorithm == "" {
		logger.Error("JWTAlgorithm is required")
		return nil, fmt.Errorf("JWTAlgorithm is required")
	}

	return &config, nil
}
