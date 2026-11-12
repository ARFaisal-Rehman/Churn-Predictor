// main.js - General application functionality

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    var anchors = document.querySelectorAll('a[href^="#"]');
    Array.prototype.forEach.call(anchors, function(anchor) {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            var targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            var targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Tooltip initialization (if needed)
    // This would typically use a library like Tippy.js or Bootstrap tooltips
});