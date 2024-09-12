package utils

import (
	"github.com/swiftyos/market/config"
	"go.uber.org/zap"
)

func NewLogger(cfg *config.Config) *zap.Logger {
	logger, _ := zap.NewProduction()
	return logger
}

func StringOrNil(s *string) string {
	if s == nil {
		return "nil"
	}
	return *s
}
