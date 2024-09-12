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
