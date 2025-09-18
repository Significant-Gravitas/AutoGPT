#!/bin/bash

# AutoGPT Platform Load Testing Script
# This script runs various k6 load tests against the AutoGPT Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default values
ENVIRONMENT=${K6_ENVIRONMENT:-"DEV"}
TEST_TYPE=${TEST_TYPE:-"load"}
VUS=${VUS:-10}
DURATION=${DURATION:-"2m"}
CLOUD_MODE=${CLOUD_MODE:-false}

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "================================================="
    echo "  AutoGPT Platform Load Testing Suite"
    echo "================================================="
    echo -e "${NC}"
}

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

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v k6 &> /dev/null; then
        print_error "k6 is not installed. Please install k6 first."
        echo "Install with: brew install k6"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq is not installed. Installing jq for JSON processing..."
        if command -v brew &> /dev/null; then
            brew install jq
        else
            print_error "Please install jq manually"
            exit 1
        fi
    fi
    
    print_success "Dependencies verified"
}

validate_environment() {
    print_info "Validating environment configuration..."
    
    # Check if environment config exists
    if [ ! -f "${SCRIPT_DIR}/configs/environment.js" ]; then
        print_error "Environment configuration not found"
        exit 1
    fi
    
    # Validate cloud configuration if cloud mode is enabled
    if [ "$CLOUD_MODE" = true ]; then
        if [ -z "$K6_CLOUD_PROJECT_ID" ] || [ -z "$K6_CLOUD_TOKEN" ]; then
            print_error "Grafana Cloud credentials not set (K6_CLOUD_PROJECT_ID, K6_CLOUD_TOKEN)"
            print_info "Run with CLOUD_MODE=false to use local mode"
            exit 1
        fi
        print_success "Grafana Cloud configuration validated"
    fi
    
    print_success "Environment validated for: $ENVIRONMENT"
}

run_load_test() {
    print_info "Running load test scenario..."
    
    local output_file="${LOG_DIR}/load_test_${TIMESTAMP}.json"
    local cloud_args=""
    
    if [ "$CLOUD_MODE" = true ]; then
        cloud_args="--out cloud"
        print_info "Running in Grafana Cloud mode"
    else
        cloud_args="--out json=${output_file}"
        print_info "Running in local mode, output: $output_file"
    fi
    
    K6_ENVIRONMENT="$ENVIRONMENT" k6 run \
        --vus "$VUS" \
        --duration "$DURATION" \
        $cloud_args \
        "${SCRIPT_DIR}/scenarios/comprehensive-platform-load-test.js"
    
    if [ "$CLOUD_MODE" = false ] && [ -f "$output_file" ]; then
        print_success "Load test completed. Results saved to: $output_file"
        
        # Generate summary
        if command -v jq &> /dev/null; then
            echo ""
            print_info "Test Summary:"
            jq -r '
                select(.type == "Point" and .metric == "http_reqs") | 
                "Total HTTP Requests: \(.data.value)"
            ' "$output_file" | tail -1
            
            jq -r '
                select(.type == "Point" and .metric == "http_req_duration") | 
                "Average Response Time: \(.data.value)ms"
            ' "$output_file" | tail -1
        fi
    else
        print_success "Load test completed and sent to Grafana Cloud"
    fi
}

run_stress_test() {
    print_info "Running stress test scenario..."
    
    local output_file="${LOG_DIR}/stress_test_${TIMESTAMP}.json"
    local cloud_args=""
    
    if [ "$CLOUD_MODE" = true ]; then
        cloud_args="--out cloud"
    else
        cloud_args="--out json=${output_file}"
    fi
    
    K6_ENVIRONMENT="$ENVIRONMENT" k6 run \
        $cloud_args \
        "${SCRIPT_DIR}/scenarios/high-concurrency-api-stress-test.js"
    
    if [ "$CLOUD_MODE" = false ] && [ -f "$output_file" ]; then
        print_success "Stress test completed. Results saved to: $output_file"
    else
        print_success "Stress test completed and sent to Grafana Cloud"
    fi
}

run_websocket_test() {
    print_info "Running WebSocket stress test..."
    
    local output_file="${LOG_DIR}/websocket_test_${TIMESTAMP}.json"
    local cloud_args=""
    
    if [ "$CLOUD_MODE" = true ]; then
        cloud_args="--out cloud"
    else
        cloud_args="--out json=${output_file}"
    fi
    
    K6_ENVIRONMENT="$ENVIRONMENT" k6 run \
        $cloud_args \
        "${SCRIPT_DIR}/scenarios/real-time-websocket-stress-test.js"
    
    if [ "$CLOUD_MODE" = false ] && [ -f "$output_file" ]; then
        print_success "WebSocket test completed. Results saved to: $output_file"
    else
        print_success "WebSocket test completed and sent to Grafana Cloud"
    fi
}

run_spike_test() {
    print_info "Running spike test..."
    
    local output_file="${LOG_DIR}/spike_test_${TIMESTAMP}.json"
    local cloud_args=""
    
    if [ "$CLOUD_MODE" = true ]; then
        cloud_args="--out cloud"
    else
        cloud_args="--out json=${output_file}"
    fi
    
    # Spike test with rapid ramp-up
    K6_ENVIRONMENT="$ENVIRONMENT" k6 run \
        --stage 10s:100 \
        --stage 30s:100 \
        --stage 10s:0 \
        $cloud_args \
        "${SCRIPT_DIR}/scenarios/comprehensive-platform-load-test.js"
    
    if [ "$CLOUD_MODE" = false ] && [ -f "$output_file" ]; then
        print_success "Spike test completed. Results saved to: $output_file"
    else
        print_success "Spike test completed and sent to Grafana Cloud"
    fi
}

show_help() {
    cat << EOF
AutoGPT Platform Load Testing Script

USAGE:
    $0 [TEST_TYPE] [OPTIONS]

TEST TYPES:
    load        Run standard load test (default)
    stress      Run stress test with high VU count
    websocket   Run WebSocket-specific stress test
    spike       Run spike test with rapid load changes
    all         Run all test scenarios sequentially

OPTIONS:
    -e, --environment ENV    Test environment (DEV, STAGING, PROD) [default: DEV]
    -v, --vus VUS           Number of virtual users [default: 10]
    -d, --duration DURATION Test duration [default: 2m]
    -c, --cloud             Run tests in Grafana Cloud mode
    -h, --help              Show this help message

EXAMPLES:
    # Run basic load test
    $0 load

    # Run stress test with 50 VUs for 5 minutes
    $0 stress -v 50 -d 5m

    # Run WebSocket test in cloud mode
    $0 websocket --cloud

    # Run all tests in staging environment
    $0 all -e STAGING

    # Run spike test with cloud reporting
    $0 spike --cloud -e DEV

ENVIRONMENT VARIABLES:
    K6_ENVIRONMENT           Target environment (DEV, STAGING, PROD)
    K6_CLOUD_PROJECT_ID      Grafana Cloud project ID
    K6_CLOUD_TOKEN           Grafana Cloud API token
    VUS                      Number of virtual users
    DURATION                 Test duration
    CLOUD_MODE               Enable cloud mode (true/false)

EOF
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--vus)
                VUS="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -c|--cloud)
                CLOUD_MODE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            load|stress|websocket|spike|all)
                TEST_TYPE="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_info "Configuration:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Test Type: $TEST_TYPE"
    echo "  Virtual Users: $VUS"
    echo "  Duration: $DURATION"
    echo "  Cloud Mode: $CLOUD_MODE"
    echo ""
    
    # Run checks
    check_dependencies
    validate_environment
    
    # Execute tests based on type
    case "$TEST_TYPE" in
        load)
            run_load_test
            ;;
        stress)
            run_stress_test
            ;;
        websocket)
            run_websocket_test
            ;;
        spike)
            run_spike_test
            ;;
        all)
            print_info "Running complete test suite..."
            run_load_test
            sleep 10  # Brief pause between tests
            run_stress_test
            sleep 10
            run_websocket_test
            sleep 10
            run_spike_test
            print_success "Complete test suite finished!"
            ;;
        *)
            print_error "Invalid test type: $TEST_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Test execution completed!"
    
    if [ "$CLOUD_MODE" = false ]; then
        print_info "Local results available in: ${LOG_DIR}/"
        print_info "To view results with Grafana Cloud, run with --cloud flag"
    else
        print_info "Results available in Grafana Cloud dashboard"
    fi
}

# Execute main function with all arguments
main "$@"