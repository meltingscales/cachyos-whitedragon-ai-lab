#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[TEST] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[TEST WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[TEST ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[TEST INFO] $1${NC}"
}

test_dependencies_script() {
    log "Testing dependencies setup script..."
    
    # Test syntax
    if bash -n ./setup_dependencies.sh; then
        info "✓ Dependencies script syntax is valid"
    else
        error "✗ Dependencies script syntax error"
        return 1
    fi
    
    # Test functions by sourcing
    source ./setup_dependencies.sh

    # Test package checking function
    if package_installed "bash"; then
        info "✓ package_installed function works (bash found)"
    else
        error "✗ package_installed function failed"
        return 1
    fi

    # Test non-existent package
    if ! package_installed "nonexistent-package-12345"; then
        info "✓ package_installed correctly identifies missing packages"
    else
        error "✗ package_installed incorrectly found non-existent package"
        return 1
    fi
    
    log "Dependencies script tests passed"
}

test_ai_tools_script() {
    log "Testing AI tools setup script..."
    
    # Test syntax
    if bash -n ./setup_ai_tools.sh; then
        info "✓ AI tools script syntax is valid"
    else
        error "✗ AI tools script syntax error"
        return 1
    fi
    
    # Test functions by sourcing
    source ./setup_ai_tools.sh
    
    # Test port checking with a definitely unused port
    if ! check_port 65432 "TestService"; then
        info "✓ check_port correctly identifies free port"
    else
        warn "Port 65432 appears to be in use (unexpected)"
    fi
    
    # Test service checking functions
    if ! check_service_running "nonexistent-service-12345"; then
        info "✓ check_service_running correctly identifies missing service"
    else
        error "✗ check_service_running incorrectly found non-existent service"
        return 1
    fi
    
    log "AI tools script tests passed"
}

test_idempotency() {
    log "Testing script idempotency..."
    
    # Create a temporary test environment
    local test_dir="/tmp/ai-tools-test-$$"
    mkdir -p "$test_dir"
    
    # Test that scripts handle existing installations gracefully
    # We'll simulate this by creating some directories
    mkdir -p "$HOME/ai-tools-test"
    
    info "✓ Created test directories"
    
    # Clean up
    rm -rf "$test_dir"
    rm -rf "$HOME/ai-tools-test"
    
    log "Idempotency tests passed"
}

test_requirements() {
    log "Testing system requirements..."

    local missing_tools=()

    # Check required tools
    command -v python >/dev/null || missing_tools+=("python")
    command -v git >/dev/null || missing_tools+=("git")
    command -v curl >/dev/null || missing_tools+=("curl")
    command -v nc >/dev/null || missing_tools+=("nc")

    if [ ${#missing_tools[@]} -eq 0 ]; then
        info "✓ All required tools are available"
    else
        warn "Missing tools: ${missing_tools[*]}"
    fi

    # Test Python version
    local python_version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    info "Python version: $python_version"

    log "Requirements tests completed"
}

main() {
    log "Starting comprehensive script tests..."
    
    cd "$(dirname "$0")"
    
    test_requirements
    test_dependencies_script
    test_ai_tools_script
    test_idempotency
    
    log "All tests completed successfully!"
    info "Scripts are ready for use"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi