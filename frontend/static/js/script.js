// Custom JavaScript for Heart Disease Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Add form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Validate that all fields are filled
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(function(field) {
                if (!field.value) {
                    isValid = false;
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
    
    // Add input validation for number fields
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(function(input) {
        input.addEventListener('blur', function() {
            if (this.min && parseFloat(this.value) < parseFloat(this.min)) {
                this.value = this.min;
            }
            if (this.max && parseFloat(this.value) > parseFloat(this.max)) {
                this.value = this.max;
            }
        });
    });
    
    // Add tooltips to form elements
    const formLabels = document.querySelectorAll('label');
    formLabels.forEach(function(label) {
        label.addEventListener('click', function() {
            const inputId = this.getAttribute('for');
            const input = document.getElementById(inputId);
            if (input) {
                input.focus();
            }
        });
    });
});