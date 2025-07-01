# Marketplace E2E Tests

This directory contains comprehensive End-to-End (E2E) tests for the AutoGPT Platform Marketplace using Playwright.

## Test Overview

The marketplace test suite covers all major functionality of the marketplace including:

- **Main Marketplace Page** (`/marketplace`)
- **Agent Detail Pages** (`/marketplace/agent/{creator}/{agent-name}`)
- **Creator Profile Pages** (`/marketplace/creator/{creator-id}`)
- **Search and Filtering Functionality**
- **Navigation and User Interactions**

## Test Files

### Core Test Files

- **`marketplace.spec.ts`** - Main marketplace page tests
  - Page load and structure validation
  - Agent and creator displays
  - Search functionality
  - Category filtering
  - Navigation tests
  - Performance and accessibility

- **`marketplace-agent.spec.ts`** - Agent detail page tests
  - Agent information display
  - Download functionality
  - Related agents
  - Creator navigation
  - Content validation

- **`marketplace-creator.spec.ts`** - Creator profile page tests
  - Creator information display
  - Creator's agents listing
  - Profile statistics
  - Navigation and interactions

- **`marketplace-search.spec.ts`** - Search and filtering tests
  - Search functionality
  - Category filtering
  - Search + filter combinations
  - Performance testing
  - Edge cases and error handling

### Page Object Models

- **`pages/marketplace.page.ts`** - Main marketplace page object
- **`pages/agent-detail.page.ts`** - Agent detail page object
- **`pages/creator-profile.page.ts`** - Creator profile page object

### Configuration

- **`marketplace.config.ts`** - Test configuration and helpers
  - Timeouts and thresholds
  - Test data and selectors
  - Helper functions
  - Performance metrics

## Running the Tests

### Prerequisites

Make sure you have the development environment running:

```bash
# Start the frontend development server
cd autogpt_platform/frontend
pnpm dev
```

The marketplace tests expect the application to be running on `http://localhost:3000`.

### Run All Marketplace Tests

```bash
# Run all marketplace tests
pnpm test marketplace

# Run with UI (headed mode)
pnpm test-ui marketplace

# Run specific test file
pnpm test marketplace.spec.ts
pnpm test marketplace-agent.spec.ts
pnpm test marketplace-creator.spec.ts
pnpm test marketplace-search.spec.ts
```

### Run Tests by Category

```bash
# Run smoke tests only
pnpm test --grep "@smoke"

# Run performance tests
pnpm test --grep "@performance"

# Run accessibility tests
pnpm test --grep "@accessibility"

# Run search-specific tests
pnpm test --grep "@search"
```

### Debug Mode

```bash
# Run in debug mode with browser visible
pnpm test marketplace.spec.ts --debug

# Run with step-by-step debugging
pnpm test marketplace.spec.ts --ui
```

## Test Structure

### Test Organization

Each test file follows this structure:

```typescript
test.describe("Feature Area", () => {
  test.beforeEach(async ({ page }) => {
    // Setup code
  });

  test.describe("Sub-feature", () => {
    test("specific functionality", async ({ page }) => {
      // Test implementation
    });
  });
});
```

### Page Object Pattern

Tests use the Page Object Model pattern for maintainability:

```typescript
// Example usage
const marketplacePage = new MarketplacePage(page);
await marketplacePage.searchAgents("Lead");
const agents = await marketplacePage.getAgentCards();
```

## Test Data

### Search Queries

- **Valid queries**: "Lead", "test", "automation", "marketing"
- **Special characters**: "@test", "#hashtag", "test!@#"
- **Edge cases**: Empty string, very long strings, non-existent terms

### Categories

- Marketing
- SEO
- Content Creation
- Automation
- Fun
- Productivity

### Test Agents

Tests work with any agents available in the marketplace, but expect at least:

- Some agents with "Lead" in the name/description
- Multiple creators with multiple agents
- Featured agents and creators

## Key Test Scenarios

### Marketplace Page Tests

1. **Page Load Validation**
   - Verify all required sections load
   - Check for proper headings and navigation
   - Validate agent cards display correctly

2. **Search Functionality**
   - Basic text search
   - Search with special characters
   - Empty and long search queries
   - Search result navigation

3. **Category Filtering**
   - Click category buttons
   - Combine search with filtering
   - Multiple category selection

4. **Agent Interactions**
   - Click agent cards
   - Navigate to agent details
   - View featured agents

5. **Creator Interactions**
   - Click creator profiles
   - Navigate to creator pages

### Agent Detail Tests

1. **Information Display**
   - Agent name, creator, description
   - Rating and run count
   - Categories and version info
   - Agent images

2. **Functionality**
   - Download button availability
   - Creator link navigation
   - Related agents display

3. **Navigation**
   - Breadcrumb navigation
   - Back to marketplace
   - Related agent navigation

### Creator Profile Tests

1. **Profile Information**
   - Creator name and handle
   - Description and statistics
   - Top categories

2. **Agent Listings**
   - Display creator's agents
   - Agent card functionality
   - Agent count accuracy

3. **Navigation**
   - Agent detail navigation
   - Back to marketplace

### Search and Filtering Tests

1. **Search Functionality**
   - Real-time search
   - Search persistence
   - Search result accuracy

2. **Category Filtering**
   - Category button responsiveness
   - Filter application
   - Filter combinations

3. **Performance**
   - Search response times
   - Filter application speed
   - UI responsiveness

## Performance Thresholds

- **Page Load**: < 15 seconds
- **Search Response**: < 5 seconds
- **Category Filtering**: < 5 seconds
- **Navigation**: < 8 seconds
- **Agent Load**: < 8 seconds

## Accessibility Testing

Tests include basic accessibility checks:

- Keyboard navigation
- ARIA attributes
- Proper heading structure
- Button and link accessibility

## Error Handling

Tests verify graceful handling of:

- Non-existent agents/creators
- Network issues
- Empty search results
- Invalid category selection
- Malformed URLs

## Test Configuration

Key configuration in `marketplace.config.ts`:

```typescript
export const MarketplaceTestConfig = {
  timeouts: {
    pageLoad: 10_000,
    navigation: 5_000,
    search: 3_000,
  },
  performance: {
    maxPageLoadTime: 15_000,
    maxSearchTime: 5_000,
  },
};
```

## Troubleshooting

### Common Issues

1. **Tests timing out**
   - Ensure the development server is running
   - Check network connectivity
   - Increase timeouts if needed

2. **Agent cards not found**
   - Verify marketplace has test data
   - Check if agent card selectors have changed
   - Look for console errors

3. **Search not working**
   - Verify search input selector
   - Check if search functionality is enabled
   - Ensure JavaScript is loaded

4. **Navigation failures**
   - Check URL patterns in config
   - Verify routing is working
   - Look for client-side errors

### Debug Tips

1. **Use headed mode** for visual debugging:

   ```bash
   pnpm test-ui marketplace.spec.ts
   ```

2. **Add debug logs** in tests:

   ```typescript
   console.log("Current URL:", page.url());
   console.log("Agent count:", agents.length);
   ```

3. **Take screenshots** on failure:

   ```typescript
   await page.screenshot({ path: "debug-screenshot.png" });
   ```

4. **Check browser console**:
   ```typescript
   page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));
   ```

## Maintenance

### Updating Tests

When marketplace UI changes:

1. Update selectors in `marketplace.config.ts`
2. Modify page object methods
3. Adjust test expectations
4. Update timeouts if needed

### Adding New Tests

1. Follow existing test structure
2. Use page object pattern
3. Add appropriate test tags
4. Include performance and accessibility checks
5. Update this README

## Contributing

When adding new marketplace tests:

1. Use descriptive test names
2. Follow the existing pattern
3. Include both positive and negative test cases
4. Add performance measurements
5. Include accessibility checks
6. Update documentation

## Test Tags

Use these tags to categorize tests:

- `@smoke` - Critical functionality
- `@regression` - Full feature testing
- `@performance` - Performance testing
- `@accessibility` - Accessibility testing
- `@search` - Search functionality
- `@filtering` - Filtering functionality
- `@navigation` - Navigation testing
- `@responsive` - Responsive design testing

Example:

```typescript
test("search functionality works @smoke @search", async ({ page }) => {
  // Test implementation
});
```
