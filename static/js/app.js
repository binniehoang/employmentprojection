// Main JavaScript for Employment Projection App

// Global variables
let currentTheme = 'light';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Employment Projection App initialized');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize theme
    initializeTheme();
    
    // Add smooth scrolling to anchor links
    initializeSmoothScrolling();
    
    // Add fade-in animation to cards
    animateCards();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize theme preferences
function initializeTheme() {
    const savedTheme = localStorage.getItem('employment-app-theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        applyTheme(currentTheme);
    }
}

// Apply theme to the application
function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    currentTheme = theme;
    localStorage.setItem('employment-app-theme', theme);
}

// Toggle between light and dark themes
function toggleTheme() {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
}

// Initialize smooth scrolling
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Animate cards on page load
function animateCards() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in-up');
        }, index * 100);
    });
}

// Utility functions

// Show loading spinner
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        const spinner = document.createElement('div');
        spinner.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="min-height: 200px;">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
        spinner.className = 'loading-spinner';
        element.appendChild(spinner);
    }
}

// Hide loading spinner
function hideLoading(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        const spinner = element.querySelector('.loading-spinner');
        if (spinner) {
            spinner.remove();
        }
    }
}

// Show success message
function showSuccess(message, duration = 5000) {
    showAlert(message, 'success', duration);
}

// Show error message
function showError(message, duration = 5000) {
    showAlert(message, 'danger', duration);
}

// Show alert message
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.querySelector('.container');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.insertBefore(alert, alertContainer.firstChild);
    
    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }
}

// Format numbers with commas
function formatNumber(num, decimals = 1) {
    if (isNaN(num)) return 'N/A';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

// Format currency
function formatCurrency(amount) {
    if (isNaN(amount)) return 'N/A';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0
    }).format(amount);
}

// Validate form data
function validateForm(formData) {
    const errors = [];
    
    for (const [key, value] of Object.entries(formData)) {
        if (value === null || value === undefined || value === '') {
            errors.push(`${key} is required`);
        }
        
        if (typeof value === 'number' && isNaN(value)) {
            errors.push(`${key} must be a valid number`);
        }
    }
    
    return errors;
}

// Handle API errors
function handleApiError(error) {
    console.error('API Error:', error);
    
    if (error.response) {
        // Server responded with error status
        const message = error.response.data?.message || error.response.statusText;
        showError(`Server Error (${error.response.status}): ${message}`);
    } else if (error.request) {
        // Request made but no response
        showError('Network Error: Unable to reach the server');
    } else {
        // Something else happened
        showError(`Error: ${error.message}`);
    }
}

// Copy text to clipboard
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showSuccess('Copied to clipboard!');
        }).catch(() => {
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

// Fallback copy function for older browsers
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showSuccess('Copied to clipboard!');
    } catch (err) {
        showError('Failed to copy to clipboard');
    }
    
    document.body.removeChild(textArea);
}

// Export functions for global use
window.EmploymentApp = {
    showLoading,
    hideLoading,
    showSuccess,
    showError,
    showAlert,
    formatNumber,
    formatCurrency,
    validateForm,
    handleApiError,
    copyToClipboard,
    toggleTheme
};