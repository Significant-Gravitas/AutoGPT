package handlers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest"
)

func TestGetPageFromContext_ValidPage(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageKey, 5)
	result := getPageFromContext(ctx)
	assert.Equal(t, 5, result)
}

func TestGetPageFromContext_InvalidPageZero(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageKey, 0)
	result := getPageFromContext(ctx)
	assert.Equal(t, 1, result)
}

func TestGetPageFromContext_NoPageValue(t *testing.T) {
	ctx := context.Background()
	result := getPageFromContext(ctx)
	assert.Equal(t, 1, result)
}

func TestGetPageFromContext_InvalidPageNegative(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageKey, -1)
	result := getPageFromContext(ctx)
	assert.Equal(t, 1, result)
}

func TestGetPageFromContext_InvalidType(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageKey, "not an int")
	result := getPageFromContext(ctx)
	assert.Equal(t, 1, result)
}

func TestGetPageSizeFromContext_ValidPageSize(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageSizeKey, 20)
	result := getPageSizeFromContext(ctx)
	assert.Equal(t, 20, result)
}

func TestGetPageSizeFromContext_InvalidPageSizeNegative(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageSizeKey, -1)
	result := getPageSizeFromContext(ctx)
	assert.Equal(t, 10, result)
}

func TestGetPageSizeFromContext_InvalidPageSizeZero(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageSizeKey, 0)
	result := getPageSizeFromContext(ctx)
	assert.Equal(t, 10, result)
}

func TestGetPageSizeFromContext_NoPageSizeValue(t *testing.T) {
	ctx := context.Background()
	result := getPageSizeFromContext(ctx)
	assert.Equal(t, 10, result)
}

func TestGetPageSizeFromContext_InvalidType(t *testing.T) {
	ctx := context.WithValue(context.Background(), pageSizeKey, "not an int")
	result := getPageSizeFromContext(ctx)
	assert.Equal(t, 10, result)
}

func TestGetNameFromContext_ValidName(t *testing.T) {
	ctx := context.WithValue(context.Background(), nameKey, "Test Name")
	result := getNameFromContext(ctx)
	assert.Equal(t, strPtr("Test Name"), result)
}

func TestGetNameFromContext_EmptyString(t *testing.T) {
	ctx := context.WithValue(context.Background(), nameKey, "")
	result := getNameFromContext(ctx)
	assert.Equal(t, strPtr(""), result)
}

func TestGetNameFromContext_NoNameValue(t *testing.T) {
	ctx := context.Background()
	result := getNameFromContext(ctx)
	assert.Nil(t, result)
}

func TestGetNameFromContext_InvalidType(t *testing.T) {
	ctx := context.WithValue(context.Background(), nameKey, 123)
	result := getNameFromContext(ctx)
	assert.Nil(t, result)
}

func TestGetKeywordsFromContext_ValidKeywords(t *testing.T) {
	ctx := context.WithValue(context.Background(), keywordsKey, "keyword1,keyword2")
	result := getKeywordsFromContext(ctx)
	assert.Equal(t, strPtr("keyword1,keyword2"), result)
}

func TestGetKeywordsFromContext_EmptyString(t *testing.T) {
	ctx := context.WithValue(context.Background(), keywordsKey, "")
	result := getKeywordsFromContext(ctx)
	assert.Equal(t, strPtr(""), result)
}

func TestGetKeywordsFromContext_NoKeywordsValue(t *testing.T) {
	ctx := context.Background()
	result := getKeywordsFromContext(ctx)
	assert.Nil(t, result)
}

func TestGetKeywordsFromContext_InvalidType(t *testing.T) {
	ctx := context.WithValue(context.Background(), keywordsKey, 123)
	result := getKeywordsFromContext(ctx)
	assert.Nil(t, result)
}

func TestGetCategoriesFromContext_ValidCategories(t *testing.T) {
	ctx := context.WithValue(context.Background(), categoriesKey, "category1,category2")
	result := getCategoriesFromContext(ctx)
	assert.Equal(t, strPtr("category1,category2"), result)
}

func TestGetCategoriesFromContext_EmptyString(t *testing.T) {
	ctx := context.WithValue(context.Background(), categoriesKey, "")
	result := getCategoriesFromContext(ctx)
	assert.Equal(t, strPtr(""), result)
}

func TestGetCategoriesFromContext_NoCategoriesValue(t *testing.T) {
	ctx := context.Background()
	result := getCategoriesFromContext(ctx)
	assert.Nil(t, result)
}

func TestGetCategoriesFromContext_InvalidType(t *testing.T) {
	ctx := context.WithValue(context.Background(), categoriesKey, 123)
	result := getCategoriesFromContext(ctx)
	assert.Nil(t, result)
}

func strPtr(s string) *string {
	return &s
}

func init() {
	logger := zaptest.NewLogger(nil)
	zap.ReplaceGlobals(logger)
}
