#!/bin/bash

# Fix Credits for Load Test Users
# This script adds credits directly to test users via database

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CREDITS_TO_ADD=10000
DATABASE_POD="autogpt-database-manager-544df49768-57xj2"
NAMESPACE="dev-agpt"

# Test user data (user_id:email)
TEST_USERS=(
    "18434a1c-34d6-4a8c-a33b-6ec0acc1490c:loadtest1@example.com"
    "58808cdf-ac94-4b69-abe8-84072e0c8d8c:loadtest2@example.com"
    "c8b18283-74e1-48e2-a43a-fc2be7f174b6:loadtest3@example.com"
    "01b9890c-c038-4eeb-be25-2a8fe3668958:stresstest1@example.com"
    "a8e72dba-8dcb-4b95-b821-8c763a6c4003:stresstest2@example.com"
)

# Functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if user exists and add credits
add_credits_to_user() {
    local user_id="$1"
    local email="$2"
    
    print_info "Adding $CREDITS_TO_ADD credits to $email ($user_id)"
    
    # SQL to add credits transaction
    local sql="
    INSERT INTO \"CreditTransaction\" (
        \"id\",
        \"userId\", 
        \"amount\", 
        \"type\", 
        \"metadata\", 
        \"createdAt\",
        \"updatedAt\"
    ) VALUES (
        gen_random_uuid(),
        '$user_id',
        $CREDITS_TO_ADD,
        'GRANT',
        '{\"reason\": \"Load testing credits\", \"automated\": true}',
        NOW(),
        NOW()
    ) ON CONFLICT DO NOTHING;
    "
    
    # Execute SQL via kubectl
    local result=$(kubectl exec -n "$NAMESPACE" "$DATABASE_POD" -- \
        psql "$DATABASE_URL" -t -c "$sql" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        print_success "Credits added successfully for $email"
        
        # Verify the user's current balance
        local balance_sql="
        SELECT COALESCE(SUM(amount), 0) as balance 
        FROM \"CreditTransaction\" 
        WHERE \"userId\" = '$user_id';
        "
        
        local balance=$(kubectl exec -n "$NAMESPACE" "$DATABASE_POD" -- \
            psql "$DATABASE_URL" -t -c "$balance_sql" 2>/dev/null | tr -d ' ')
        
        if [[ -n "$balance" && "$balance" =~ ^[0-9]+$ ]]; then
            print_success "Current balance for $email: $balance credits"
        fi
    else
        print_error "Failed to add credits for $email: $result"
    fi
}

# Check database connectivity
check_database() {
    print_info "Checking database connectivity..."
    
    local test_sql="SELECT 1 as test;"
    local result=$(kubectl exec -n "$NAMESPACE" "$DATABASE_POD" -- \
        psql "$DATABASE_URL" -t -c "$test_sql" 2>&1)
    
    if [[ $? -eq 0 && "$result" =~ "1" ]]; then
        print_success "Database connection successful"
        return 0
    else
        print_error "Database connection failed: $result"
        return 1
    fi
}

# Verify users exist
verify_users() {
    print_info "Verifying test users exist in database..."
    
    for user_data in "${TEST_USERS[@]}"; do
        local user_id="${user_data%%:*}"
        local email="${user_data##*:}"
        
        local check_sql="SELECT id, email FROM \"User\" WHERE id = '$user_id';"
        local result=$(kubectl exec -n "$NAMESPACE" "$DATABASE_POD" -- \
            psql "$DATABASE_URL" -t -c "$check_sql" 2>/dev/null)
        
        if [[ -n "$result" && "$result" =~ "$user_id" ]]; then
            print_success "User verified: $email"
        else
            print_warning "User not found in database: $email ($user_id)"
        fi
    done
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "================================================="
    echo "  Fix Credits for Load Test Users"
    echo "================================================="
    echo -e "${NC}"
    
    print_info "Adding $CREDITS_TO_ADD credits to ${#TEST_USERS[@]} test users"
    echo ""
    
    # Check database connectivity
    if ! check_database; then
        print_error "Cannot proceed without database access"
        exit 1
    fi
    
    # Verify users exist
    verify_users
    echo ""
    
    # Add credits to each user
    for user_data in "${TEST_USERS[@]}"; do
        local user_id="${user_data%%:*}"
        local email="${user_data##*:}"
        add_credits_to_user "$user_id" "$email"
        sleep 1  # Brief pause between operations
    done
    
    echo ""
    print_success "Credit addition completed!"
    
    # Show summary
    print_info "Summary of credit balances:"
    for user_data in "${TEST_USERS[@]}"; do
        local user_id="${user_data%%:*}"
        local email="${user_data##*:}"
        local balance_sql="SELECT COALESCE(SUM(amount), 0) FROM \"CreditTransaction\" WHERE \"userId\" = '$user_id';"
        local balance=$(kubectl exec -n "$NAMESPACE" "$DATABASE_POD" -- \
            psql "$DATABASE_URL" -t -c "$balance_sql" 2>/dev/null | tr -d ' ')
        
        if [[ -n "$balance" && "$balance" =~ ^[0-9]+$ ]]; then
            echo "  $email: $balance credits"
        fi
    done
    
    echo ""
    print_success "Test users are now ready for load testing!"
    print_info "You can now run: ./run-tests.sh load --cloud"
}

# Execute main function
main "$@"