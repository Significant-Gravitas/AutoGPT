#!/bin/bash

# AutoGPT Platform Test User Setup Script
# This script helps create and configure test users for load testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPABASE_URL=${SUPABASE_URL:-"https://adfjtextkuilwuhzdjpf.supabase.co"}
ENVIRONMENT=${K6_ENVIRONMENT:-"DEV"}
# Use actual dev instance service role key
DEFAULT_SERVICE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFkZmp0ZXh0a3VpbHd1aHpkanBmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMDI1MTcwMiwiZXhwIjoyMDQ1ODI3NzAyfQ.JHadCgyuMVejDxl66DIe4ZlB1ra7IGDLEkABhSJm540"
SUPABASE_SERVICE_ROLE_KEY=${SUPABASE_SERVICE_ROLE_KEY:-$DEFAULT_SERVICE_KEY}

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

print_header() {
    echo -e "${BLUE}"
    echo "================================================="
    echo "  AutoGPT Platform Test User Setup"
    echo "================================================="
    echo -e "${NC}"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found, installing..."
        if command -v brew &> /dev/null; then
            brew install jq
        else
            print_error "Please install jq manually"
            exit 1
        fi
    fi
    
    print_success "Dependencies verified"
}

create_test_user() {
    local email="$1"
    local password="$2"
    local description="$3"
    
    print_info "Creating test user: $email"
    
    # Create user via Supabase Auth API
    local response=$(curl -s -X POST \
        "${SUPABASE_URL}/auth/v1/signup" \
        -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"email\": \"$email\",
            \"password\": \"$password\",
            \"data\": {
                \"description\": \"$description\",
                \"test_user\": true,
                \"created_for\": \"load_testing\"
            }
        }")
    
    local status_code=$(echo "$response" | jq -r '.error.status // 200')
    
    if [ "$status_code" -eq 200 ] || [ "$status_code" -eq 201 ]; then
        local user_id=$(echo "$response" | jq -r '.user.id // .id')
        print_success "User created successfully: $user_id"
        
        # Add initial credits for testing
        add_test_credits "$user_id" "$email"
        
        return 0
    else
        local error_message=$(echo "$response" | jq -r '.error.message // .message // "Unknown error"')
        
        # Check if user already exists
        if [[ "$error_message" == *"already registered"* ]] || [[ "$error_message" == *"already exists"* ]]; then
            print_warning "User already exists: $email"
            return 0
        else
            print_error "Failed to create user $email: $error_message"
            return 1
        fi
    fi
}

add_test_credits() {
    local user_id="$1"
    local email="$2"
    local credits=${TEST_CREDITS:-1000}
    
    print_info "Adding $credits test credits for $email"
    
    # This would typically require admin API access
    # For now, just log the requirement
    print_warning "Manual step required: Add $credits credits to user $user_id via admin panel"
    
    # If you have an admin API endpoint for adding credits:
    # curl -s -X POST \
    #     "${API_BASE_URL}/admin/credits/add" \
    #     -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    #     -H "Content-Type: application/json" \
    #     -d "{\"user_id\": \"$user_id\", \"amount\": $credits}"
}

verify_test_user() {
    local email="$1"
    local password="$2"
    
    print_info "Verifying test user: $email"
    
    # Try to authenticate the user
    local auth_response=$(curl -s -X POST \
        "${SUPABASE_URL}/auth/v1/token?grant_type=password" \
        -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"email\": \"$email\",
            \"password\": \"$password\"
        }")
    
    local access_token=$(echo "$auth_response" | jq -r '.access_token // empty')
    
    if [ -n "$access_token" ]; then
        print_success "User authentication verified: $email"
        
        # Test API access
        local api_response=$(curl -s -X POST \
            "${API_BASE_URL}/api/v1/auth/user" \
            -H "Authorization: Bearer $access_token" \
            -H "Content-Type: application/json" \
            -d '{}')
        
        local user_id=$(echo "$api_response" | jq -r '.id // empty')
        
        if [ -n "$user_id" ]; then
            print_success "API access verified for user: $user_id"
            return 0
        else
            print_error "API access failed for $email"
            return 1
        fi
    else
        local error_message=$(echo "$auth_response" | jq -r '.error.message // .message // "Authentication failed"')
        print_error "Authentication failed for $email: $error_message"
        return 1
    fi
}

setup_test_users() {
    print_info "Setting up test users from configuration..."
    
    local config_file="${SCRIPT_DIR}/data/test-users.json"
    
    if [ ! -f "$config_file" ]; then
        print_error "Test users configuration not found: $config_file"
        exit 1
    fi
    
    # Read test users from configuration
    local test_users=$(jq -r '.test_users[] | @base64' "$config_file")
    local success_count=0
    local total_count=0
    
    for user_data in $test_users; do
        local user=$(echo "$user_data" | base64 --decode)
        local email=$(echo "$user" | jq -r '.email')
        local password=$(echo "$user" | jq -r '.password')
        local description=$(echo "$user" | jq -r '.description')
        
        total_count=$((total_count + 1))
        
        if create_test_user "$email" "$password" "$description"; then
            success_count=$((success_count + 1))
        fi
        
        # Brief pause between user creation
        sleep 1
    done
    
    print_info "Test user setup complete: $success_count/$total_count users processed"
    
    if [ $success_count -eq $total_count ]; then
        print_success "All test users set up successfully!"
    else
        print_warning "Some test users may need manual setup"
    fi
}

verify_all_test_users() {
    print_info "Verifying all test users..."
    
    local config_file="${SCRIPT_DIR}/data/test-users.json"
    local test_users=$(jq -r '.test_users[] | @base64' "$config_file")
    local success_count=0
    local total_count=0
    
    for user_data in $test_users; do
        local user=$(echo "$user_data" | base64 --decode)
        local email=$(echo "$user" | jq -r '.email')
        local password=$(echo "$user" | jq -r '.password')
        
        total_count=$((total_count + 1))
        
        if verify_test_user "$email" "$password"; then
            success_count=$((success_count + 1))
        fi
        
        sleep 1
    done
    
    print_info "Verification complete: $success_count/$total_count users verified"
    
    if [ $success_count -eq $total_count ]; then
        print_success "All test users verified and ready for load testing!"
    else
        print_error "Some test users need attention before load testing"
        exit 1
    fi
}

cleanup_test_data() {
    print_info "Cleaning up test data..."
    
    print_warning "This will remove test graphs, executions, and other test data"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleanup cancelled"
        exit 0
    fi
    
    # This would require admin API access to clean up test data
    print_warning "Manual cleanup required:"
    echo "1. Remove test graphs created by test users"
    echo "2. Clean up test executions and schedules"
    echo "3. Reset test user credits if needed"
    echo "4. Clear any test integrations or API keys"
    
    print_success "Cleanup reminder provided"
}

show_test_user_status() {
    print_info "Test User Status Summary"
    echo ""
    
    local config_file="${SCRIPT_DIR}/data/test-users.json"
    local test_users=$(jq -r '.test_users[] | @base64' "$config_file")
    
    echo "Environment: $ENVIRONMENT"
    echo "Supabase URL: $SUPABASE_URL"
    echo "API URL: ${API_BASE_URL:-"Not set"}"
    echo ""
    echo "Configured Test Users:"
    
    for user_data in $test_users; do
        local user=$(echo "$user_data" | base64 --decode)
        local email=$(echo "$user" | jq -r '.email')
        local description=$(echo "$user" | jq -r '.description')
        
        echo "  - $email ($description)"
    done
    
    echo ""
    print_info "Run './setup-test-users.sh verify' to check user access"
}

show_help() {
    cat << EOF
AutoGPT Platform Test User Setup Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    setup       Create test users from configuration
    verify      Verify test user authentication and API access
    status      Show test user configuration summary
    cleanup     Clean up test data (manual steps)
    help        Show this help message

OPTIONS:
    -e, --environment ENV    Target environment (DEV, STAGING, PROD)
    -c, --credits CREDITS    Initial credits for test users [default: 1000]

ENVIRONMENT VARIABLES:
    SUPABASE_URL             Supabase instance URL
    SUPABASE_SERVICE_ROLE_KEY Service role key for user creation
    API_BASE_URL             API base URL for testing
    K6_ENVIRONMENT           Target environment
    TEST_CREDITS             Initial credits for test users

EXAMPLES:
    # Set up all test users
    $0 setup

    # Verify test users can authenticate
    $0 verify

    # Show current configuration
    $0 status

    # Clean up test data
    $0 cleanup

PREREQUISITES:
    1. Set SUPABASE_SERVICE_ROLE_KEY environment variable
    2. Configure test users in data/test-users.json
    3. Ensure target environment is accessible

EOF
}

# Parse command line arguments
COMMAND=${1:-"help"}
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--credits)
            TEST_CREDITS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set environment-specific variables
case "$ENVIRONMENT" in
    DEV)
        API_BASE_URL="https://dev-api.agpt.co"
        ;;
    STAGING)
        API_BASE_URL="https://staging-api.agpt.co"
        ;;
    PROD)
        API_BASE_URL="https://api.agpt.co"
        ;;
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Main execution
main() {
    print_header
    
    case "$COMMAND" in
        setup)
            check_dependencies
            
            if [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
                print_error "SUPABASE_SERVICE_ROLE_KEY environment variable is required"
                exit 1
            fi
            
            setup_test_users
            ;;
        verify)
            check_dependencies
            verify_all_test_users
            ;;
        status)
            show_test_user_status
            ;;
        cleanup)
            cleanup_test_data
            ;;
        help)
            show_help
            ;;
        *)
            print_error "Invalid command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main