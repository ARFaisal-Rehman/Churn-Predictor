// profile.js - Handles customer profile page interactions

document.addEventListener('DOMContentLoaded', function() {
    // Add animations to profile cards
    var cards = document.querySelectorAll('.card');
    
    // Stagger the card animations
    Array.prototype.forEach.call(cards, function(card, index) {
        setTimeout(function() {
            card.classList.add('visible');
        }, 100 * index);
    });
    
    // Add functionality to offer action buttons
    var modifyOfferBtn = document.querySelector('.offer-actions .btn-secondary');
    var applyOfferBtn = document.querySelector('.offer-actions .btn');
    
    if (modifyOfferBtn) {
        modifyOfferBtn.addEventListener('click', function() {
            // In a real app, this would open a modal to modify the offer
            // For demo purposes, we'll just show an alert
            alert('In a full implementation, this would open a modal to customize the retention offer');
        });
    }
    
    if (applyOfferBtn) {
        applyOfferBtn.addEventListener('click', function() {
            // In a real app, this would apply the offer to the customer
            // For demo purposes, we'll just show a success message
            var offerContainer = document.querySelector('.offer-container');
            
            // Show success animation
            offerContainer.classList.add('offer-applied');
            
            // Change button text
            this.textContent = 'Offer Applied';
            this.disabled = true;
            
            // Add success message
            var successMsg = document.createElement('div');
            successMsg.className = 'success-message';
            successMsg.innerHTML = '<i class="fas fa-check-circle"></i> Offer successfully applied';
            
            // Insert after offer container
            offerContainer.parentNode.insertBefore(successMsg, offerContainer.nextSibling);
        });
    }
});

// Add CSS for animations
document.addEventListener('DOMContentLoaded', function() {
    var style = document.createElement('style');
    style.textContent = 
        '.card {' +
        '    opacity: 0;' +
        '    transform: translateY(20px);' +
        '    transition: opacity 0.5s ease, transform 0.5s ease;' +
        '}' +
        
        '.card.visible {' +
        '    opacity: 1;' +
        '    transform: translateY(0);' +
        '}' +
        
        '.offer-container {' +
        '    transition: all 0.3s ease;' +
        '}' +
        
        '.offer-container.offer-applied {' +
        '    transform: scale(1.02);' +
        '    box-shadow: 0 0 15px rgba(6, 214, 160, 0.3);' +
        '}' +
        
        '.success-message {' +
        '    background-color: rgba(6, 214, 160, 0.1);' +
        '    color: var(--low-risk);' +
        '    padding: var(--spacing-md);' +
        '    border-radius: var(--border-radius-sm);' +
        '    margin: var(--spacing-md) 0;' +
        '    display: flex;' +
        '    align-items: center;' +
        '    animation: fadeIn 0.5s ease;' +
        '}' +
        
        '.success-message i {' +
        '    margin-right: var(--spacing-sm);' +
        '}' +
        
        '@keyframes fadeIn {' +
        '    from { opacity: 0; transform: translateY(-10px); }' +
        '    to { opacity: 1; transform: translateY(0); }' +
        '}' +
        
        '.btn:disabled {' +
        '    background-color: var(--low-risk);' +
        '    cursor: not-allowed;' +
        '}';
    document.head.appendChild(style);
});

// Initialize Gauge Chart for Churn Probability
document.addEventListener('DOMContentLoaded', function() {
    try {
        var canvas = document.getElementById('gaugeChart');
        if (!canvas) {
            // It's possible we're not on the profile page, or the element is missing
            return;
        }
        
        var ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error('Could not get 2d context from canvas');
            return;
        }
        
        try {
            // Parse embedded customer JSON to avoid template syntax inside JS
            var cEl = document.getElementById('customer-data');
            if (!cEl) {
                console.warn('Customer data template not found');
                return;
            }
            
            var customerData = JSON.parse(cEl.textContent || '{}');
            var churnProbability = Math.round((customerData.churn_probability || 0) * 100);
            var remainingProbability = 100 - churnProbability;
            
            // Determine background color based on risk category
            var riskColor = '#06d6a0';
            if ((customerData.risk_category || '').toLowerCase() === 'high') {
                riskColor = '#ff6b6b';
            } else if ((customerData.risk_category || '').toLowerCase() === 'medium') {
                riskColor = '#ffd166';
            }
            
            var gaugeChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [churnProbability, remainingProbability],
                        backgroundColor: [
                            riskColor,
                            '#f8f9fa'
                        ],
                        borderWidth: 0,
                        cutout: '80%'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    rotation: -90,
                    circumference: 180,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            enabled: false
                        }
                    }
                },
                plugins: [{
                    id: 'gaugeText',
                    afterDraw: function(chart) {
                        var width = chart.width;
                        var height = chart.height;
                        var ctx = chart.ctx;
                        
                        ctx.restore();
                        ctx.font = '1.5rem Roboto';
                        ctx.textBaseline = 'middle';
                        ctx.textAlign = 'center';
                        
                        var text = churnProbability + '%';
                        var textX = width / 2;
                        var textY = height - 30;
                        
                        ctx.fillText(text, textX, textY);
                        ctx.save();
                    }
                }]
            });
        } catch (chartError) {
            console.error('Error creating gauge chart:', chartError);
        }
    } catch (error) {
        console.error('Error initializing gauge chart:', error);
    }
});