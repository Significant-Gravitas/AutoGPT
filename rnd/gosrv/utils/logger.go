package utils

import (
	"github.com/swiftyos/market/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"os"
)

func NewLogger(cfg *config.Config) *zap.Logger {
	encoderConfig := zap.NewProductionEncoderConfig()
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	consoleEncoder := zapcore.NewConsoleEncoder(encoderConfig)
	consoleWriter := zapcore.AddSync(os.Stdout)
	core := zapcore.NewCore(consoleEncoder, consoleWriter, zap.InfoLevel)

	logger := zap.New(core)
	return logger
}

func StringOrNil(s *string) string {
	if s == nil {
		return "nil"
	}
	return *s
}
