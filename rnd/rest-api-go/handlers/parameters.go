package handlers

import (
	"context"
	"strconv"
)

const pageKey string = "page"

func getPageFromContext(ctx context.Context) int {
	defaultPage := 1
	if ctx == nil {
		return defaultPage
	}

	pageValue := ctx.Value(pageKey)

	if pageValue == nil {
		return defaultPage
	}

	// Type assertion to check if the value is an int
	if page, ok := pageValue.(int); ok {
		if page < 1 {
			return defaultPage
		}
		return page
	}

	// If it's not an int, try to convert from string
	if pageStr, ok := pageValue.(string); ok {
		page, err := strconv.Atoi(pageStr)
		if err != nil || page < 1 {
			return defaultPage
		}
		return page
	}

	return defaultPage
}

const pageSizeKey string = "page_size"

func getPageSizeFromContext(ctx context.Context) int {
	pageSizeValue := ctx.Value(pageSizeKey)
	if pageSizeValue == nil {
		return 10
	}
	if pageSizeValue, ok := pageSizeValue.(int); ok {
		if pageSizeValue < 1 {
			return 10
		}
		return pageSizeValue
	}
	return 10
}

const nameKey string = "name"

func getNameFromContext(ctx context.Context) *string {
	nameValue := ctx.Value(nameKey)
	if nameValue == nil {
		return nil
	}
	if nameValue, ok := nameValue.(string); ok {
		return &nameValue
	}
	return nil
}

const keywordsKey string = "keywords"

func getKeywordsFromContext(ctx context.Context) *string {
	keywordsValue := ctx.Value(keywordsKey)
	if keywordsValue == nil {
		return nil
	}
	if keywordsValue, ok := keywordsValue.(string); ok {
		return &keywordsValue
	}
	return nil
}

const categoriesKey string = "categories"

func getCategoriesFromContext(ctx context.Context) *string {
	categoriesValue := ctx.Value(categoriesKey)
	if categoriesValue == nil {
		return nil
	}
	if categoriesValue, ok := categoriesValue.(string); ok {
		return &categoriesValue
	}
	return nil
}
