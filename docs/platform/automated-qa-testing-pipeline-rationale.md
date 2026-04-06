# Automated QA Testing Pipeline Rationale

This document explains the reasoning behind the proposed automated QA testing pipeline work for release and PR validation.

It is meant to answer:

- Why are we changing the current E2E approach?
- Why are we splitting tests into different tiers?
- Why should some flows block PRs while others should not?
- Why should we invest in seed data, auth reuse, and reporting before simply adding more tests?

## Problem

The current release checklist is much broader than the current deterministic Playwright coverage.

At the moment, the existing frontend Playwright suite covers only a small portion of the release checklist end-to-end:

- around `6/37` main checklist items are fully covered
- around `11/37` are only partially covered
- `0/19` AutoPilot checklist items are meaningfully covered as user workflows

That means the current E2E suite is useful, but it is not yet a reliable PR-gating replacement for manual release QA.

## Why The Current Approach Is Not Enough

### 1. The current suite is not organized around merge confidence

The goal of this project is not simply to "have more E2E tests". The goal is to increase confidence that a PR is safe to merge.

Those are different goals:

- A large suite can still be poor at merge confidence if it is slow, flaky, or checking low-value paths.
- A smaller suite can provide better merge confidence if it validates the most critical product flows reliably on every PR.

The current test suite has valuable coverage, but it was not designed as a clearly scoped PR smoke suite for release-critical flows.

### 2. The current setup does expensive work through the UI

Today the Playwright setup creates users through the signup UI in `src/tests/global-setup.ts`.

That has multiple drawbacks:

- it adds runtime before tests even begin
- it duplicates auth/onboarding work that should be seeded once
- it increases flake because the setup itself depends on the product UI
- when setup fails, the entire suite becomes noisy even if the product being tested is fine

Using the UI to prepare state is acceptable for testing signup itself, but it is not a good default for preparing every other test.

### 3. The suite repeats login and setup work too often

Many tests log in through the UI in `beforeEach`. This is simple, but expensive.

That means we repeatedly pay for:

- page loads
- auth redirects
- onboarding skips
- post-login stabilization

When the suite grows, repeated setup work becomes one of the biggest contributors to runtime and flake.

### 4. Not all checklist items belong in the same test category

The release checklist mixes together several different types of work:

- deterministic product UI flows
- flows that depend on mail, Stripe, external auth, or admin tools
- operational and manual checks
- AI-driven AutoPilot behaviors

Treating all of these as one flat PR-blocking E2E suite would produce a system that is too slow, too fragile, and too expensive.

### 5. AutoPilot behavior is not yet a good default merge gate

AutoPilot flows are valuable, but they are fundamentally different from deterministic UI tests.

They depend on:

- model output quality
- tool-choice behavior
- external web results
- streaming timing
- prompt sensitivity

That makes them good candidates for exploratory or nightly validation, but poor candidates for the first wave of hard PR-blocking tests.

## Proposed Direction

The proposed implementation is built around one principle:

> PR checks should optimize for reliable merge confidence, not checklist completeness in a single job.

That leads to the following design.

## Step 1: Split The Test System Into Tiers

### Proposal

Create three test tiers:

- `PR smoke`: small, deterministic, blocking
- `Release regression`: broader checklist coverage, nightly or manually triggered
- `AI exploratory`: AutoPilot and non-deterministic workflows, non-blocking

### Why

Different flows have different reliability and maintenance profiles.

If we keep everything in one bucket:

- PRs become slower
- flake blocks merges
- engineers stop trusting the suite
- costs rise as the suite grows

If we split by purpose:

- PR jobs stay fast and trustworthy
- broader coverage can still exist without hurting developer velocity
- AI-based and external-provider tests can still provide signal without blocking merges

### Why Not Keep One Big Suite

Because one big suite forces a bad tradeoff:

- either it becomes too small to be useful
- or it becomes too noisy to trust

The tiered model is the standard way to avoid that trap.

## Step 2: Move Setup From The UI Into Seeded Test State

### Proposal

Seed fixed QA users, agents, marketplace data, creator data, and admin data in backend test fixtures instead of generating most of it through the frontend UI.

### Why

Seeded state is:

- faster
- more deterministic
- easier to debug
- easier to make role-specific

We should still test signup, onboarding, and similar flows through the UI when those are the actual things under test. But we should not use them as the default preparation step for unrelated tests.

### Why Not Continue Creating Users In Global Setup

Because it makes the whole suite pay a setup tax for every run:

- extra time
- extra flake
- extra hidden coupling between tests

UI-created setup is good for validating signup. It is bad as baseline infrastructure for the rest of the suite.

## Step 3: Reuse Auth State Instead Of Re-Logging In Repeatedly

### Proposal

Create persistent Playwright `storageState` files for seeded accounts and use role-based fixtures:

- regular user
- builder user
- library user
- creator user
- admin user
- billing/reset-password user

### Why

Most tests do not need to validate the login form itself.

If a test is really about builder, library, marketplace, or profile behavior, then repeated UI login is wasted work.

Reusing auth state:

- cuts runtime
- reduces redirect-related flake
- keeps tests focused on their real purpose

### Why Not Log In Through The UI In Every Test

Because that makes test intent less clear and adds cost without adding coverage.

Only login-specific tests should pay for login-specific setup by default.

## Step 4: Create A Dedicated PR Smoke Suite

### Proposal

Create a small `@smoke` suite that runs on every PR and contains only the highest-value deterministic flows.

Suggested phase 1 smoke candidates:

- multi-tab logout
- signup
- onboarding
- builder tutorial
- build a basic agent
- run an agent
- schedule an agent
- import an agent
- open an imported agent in builder
- marketplace add-to-library
- marketplace download
- profile save
- notification persistence

### Why

These are the flows most likely to increase trust at merge time.

They cover the product areas most visible to users:

- auth
- first-run experience
- core builder flow
- run flow
- library flow
- marketplace flow
- account settings

### Why Not Put The Full Checklist In The PR Suite

Because some checklist items are:

- operational, not UI tests
- dependent on external systems
- admin-only
- AI-driven
- hard to keep deterministic

A PR-blocking suite must prioritize speed and signal quality over completeness.

## Step 5: Put Broader Release Coverage In A Nightly Or Manual Regression Suite

### Proposal

Create a broader release/regression suite for checklist items that are valuable, but not good first-wave PR blockers.

Examples:

- forgot password
- login with new password
- full task lifecycle
- export flow
- saved-agent edit-and-run chains
- marketplace submission success
- creator dashboard cleanup
- admin approval and revocation
- credit validation
- Stripe test-card flow

### Why

These flows still matter. They just have a worse reliability-to-cost ratio for every single PR.

Running them nightly or on demand gives us broader product confidence without slowing down every merge.

### Why Not Ignore These Flows Entirely

Because they still represent real release risk.

The answer is not to skip them. The answer is to run them in the right tier.

## Step 6: Keep AutoPilot In A Separate Non-Blocking Suite First

### Proposal

Run AutoPilot workflows in a dedicated exploratory or nightly suite instead of in the first wave of blocking PR checks.

Examples:

- sending a chat message and getting a response
- web search
- tool or block execution
- create/edit/schedule agent through chat
- file upload context
- voice
- thread continuity
- graceful recovery after failure

### Why

These flows are important, but much less deterministic than classic UI paths.

They depend on:

- model output variance
- tool selection behavior
- streaming timing
- prompt quality
- external systems

If they fail, it is often harder to know whether the issue is:

- a real regression
- an infra issue
- model variance
- bad prompt routing

That makes them poor first-wave merge blockers.

### Why Not Block PRs On AutoPilot Immediately

Because a flaky merge gate reduces trust much faster than it increases coverage.

AutoPilot should still be tested, but not forced into the same reliability expectations as deterministic Playwright smoke tests on day one.

## Step 7: Add Better Reporting And Flake Tracking

### Proposal

For the PR smoke suite:

- retry failures up to `2` times
- upload screenshots/videos/traces on failure
- generate a summary per flow
- post a PR comment with pass/fail per flow
- track retry passes as flake signals

### Why

Without reporting, the suite becomes harder to trust and harder to improve.

We need to know:

- which flow failed
- whether it failed once or repeatedly
- whether it passed on retry
- what artifact to inspect

That makes the suite useful both as a merge gate and as a quality feedback loop.

### Why Not Just Rely On Raw Playwright Output

Because raw output is too low-level for a release checklist workflow.

The ticket requires flow-level pass/fail signals and failure artifacts that are easy to consume from the PR.

## Step 8: Prefer Integration Tests For Broader Frontend Coverage

### Proposal

Use page-level integration tests for broader frontend logic and reserve E2E for critical happy paths.

### Why

E2E is expensive:

- slower to run
- more brittle
- more dependent on environment quality

Integration tests are better for:

- page behavior
- API response handling
- UI states
- edge cases

That lets us keep E2E focused on what really needs a browser and a full stack.

### Why Not Solve All Coverage Gaps With E2E

Because that increases cost and maintenance faster than it increases confidence.

E2E should be used where it is uniquely valuable, not as the default answer to every missing test.

## What This Means For The Current Repo

The current repo already has useful Playwright coverage, but it needs to be reshaped for PR-gating use.

The main changes are not just "write more tests". They are:

- reclassify tests by purpose
- reduce setup cost
- reduce repeated auth/setup work
- improve deterministic seeded state
- add a proper smoke suite
- add flow-level reporting

## Why This Is Better Than The Current Way

The current direction of simply extending the existing suite without changing its structure would likely lead to:

- longer runtimes
- more setup cost
- more flaky failures
- more expensive CI
- less trust in merge gates

The proposed direction gives us:

- faster PR checks
- stronger signal on the most important flows
- room to expand coverage safely
- lower flake pressure
- a path for AutoPilot testing without making PR checks unstable

## Expected Outcome

If implemented in this order, the test system should become:

- more useful for merge confidence
- easier to maintain
- cheaper per PR than a naive "run everything" model
- better aligned with the actual release checklist

The key idea is simple:

> We are not optimizing for maximum test count.
> We are optimizing for trustworthy automated confidence at merge time.
