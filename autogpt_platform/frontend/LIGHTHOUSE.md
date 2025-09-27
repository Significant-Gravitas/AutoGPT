# Lighthouse CI Setup

This document describes the Google Lighthouse CI integration for the AutoGPT Platform frontend.

## Overview

Google Lighthouse is automatically run on the CI for every push to `master` or `dev` branches and on pull requests. It audits the application's performance, accessibility, SEO, and best practices.

## Configuration

### Main Configuration (`lighthouserc.json`)

- **URLs audited**: `/`, `/build`, `/login`
- **Number of runs**: 3 per URL (for consistent results)
- **Thresholds**:
  - Performance: 70% (warn)
  - Accessibility: 90% (error)
  - Best Practices: 85% (warn)
  - SEO: 85% (warn)
  - PWA: Disabled

### Local Testing (`lighthouserc.local.json`)

- Lower thresholds for development testing
- Single run per URL for faster feedback

## Usage

### CI Integration

Lighthouse runs automatically in the `lighthouse` job of the `platform-frontend-ci.yml` workflow. Reports are uploaded as CI artifacts.

### Local Development

```bash
# Start the development server
pnpm dev

# In another terminal, run Lighthouse (with more lenient thresholds)
pnpm lighthouse:local
```

### Production Testing

```bash
# Build and start the application
pnpm build && pnpm start

# In another terminal, run full Lighthouse audit
pnpm lighthouse
```

## Reports

- **CI**: Reports are stored as artifacts and can be downloaded from the GitHub Actions run page
- **Local**: Reports are saved to `lhci_reports/` directory
- **Format**: Both HTML and JSON reports are generated

## Customization

To modify the audit configuration:

1. **URLs**: Edit the `collect.url` array in `lighthouserc.json`
2. **Thresholds**: Adjust the `assert.assertions` values
3. **Categories**: Add or remove audit categories as needed

## Troubleshooting

### Common Issues

1. **Frontend not ready**: Ensure the application is fully started before running Lighthouse
2. **Memory issues**: Lighthouse can be memory-intensive; ensure sufficient resources
3. **Network timeouts**: Check that all services are healthy before running audits

### Debug Commands

```bash
# Check configuration validity
pnpm lhci healthcheck

# Run only collection (no assertions)
pnpm lhci collect

# Run only assertions on existing reports
pnpm lhci assert
```

## Performance Guidelines

- **Performance scores below 70%** will generate warnings
- **Accessibility scores below 90%** will fail the CI
- Focus on critical user paths for consistent performance
- Consider performance budgets for key metrics

## Integration with Development Workflow

1. **Pre-merge**: Lighthouse runs on all PRs to catch regressions
2. **Post-merge**: Results on `master`/`dev` establish performance baselines
3. **Monitoring**: Track performance trends over time via CI artifacts
