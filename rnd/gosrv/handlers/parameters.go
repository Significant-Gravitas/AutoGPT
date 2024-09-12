package handlers

import (
	"context"

	"go.uber.org/zap"
)

const pageKey string = "page"

func getPageFromContext(ctx context.Context) int {
	if pageValue, ok := ctx.Value(pageKey).(int); ok {
		if pageValue < 1 {
			zap.L().Error("Invalid page value", zap.Int("page", pageValue))
			return 1
		}
		return pageValue
	}
	return 1
}

const pageSizeKey string = "page_size"

func getPageSizeFromContext(ctx context.Context) int {
	if pageSizeValue, ok := ctx.Value(pageSizeKey).(int); ok {
		if pageSizeValue < 1 {
			zap.L().Error("Invalid page size value", zap.Int("page_size", pageSizeValue))
			return 10
		}
		return pageSizeValue
	}
	return 10
}

const nameKey string = "name"

func getNameFromContext(ctx context.Context) *string {
	if nameValue, ok := ctx.Value(nameKey).(string); ok {
		zap.L().Debug("Retrieved name from context", zap.String("name", nameValue))
		return &nameValue
	}
	zap.L().Debug("No name found in context")
	return nil
}

const keywordsKey string = "keywords"

func getKeywordsFromContext(ctx context.Context) *string {
	if keywordsValue, ok := ctx.Value(keywordsKey).(string); ok {
		zap.L().Debug("Retrieved keywords from context", zap.String("keywords", keywordsValue))
		return &keywordsValue
	}
	zap.L().Debug("No keywords found in context")
	return nil
}

const categoriesKey string = "categories"

func getCategoriesFromContext(ctx context.Context) *string {
	if categoriesValue, ok := ctx.Value(categoriesKey).(string); ok {
		zap.L().Debug("Retrieved categories from context", zap.String("categories", categoriesValue))
		return &categoriesValue
	}
	zap.L().Debug("No categories found in context")
	return nil
}