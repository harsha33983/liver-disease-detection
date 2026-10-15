/**
 * LiverCare AI - Main Client-Side Logic
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Sidebar Toggle
  const sidebar = document.getElementById('app-sidebar');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebarBackdrop = document.getElementById('sidebar-backdrop');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', function () {
      sidebar.classList.toggle('show');
      if (sidebarBackdrop) {
        sidebarBackdrop.classList.toggle('d-none');
      }
    });
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', function () {
      sidebar.classList.remove('show');
      sidebarBackdrop.classList.add('d-none');
    });
  }

  // 2. Form Submission Loading Overlay
  const predictionForm = document.getElementById('prediction-form');
  const loadingOverlay = document.getElementById('loading-overlay');

  if (predictionForm && loadingOverlay) {
    predictionForm.addEventListener('submit', function (e) {
      if (predictionForm.checkValidity()) {
        loadingOverlay.style.display = 'flex';
      }
    });
  }

  // 3. Reset Button Confirmation
  const resetBtn = document.getElementById('btn-reset-form');
  if (resetBtn && predictionForm) {
    resetBtn.addEventListener('click', function (e) {
      if (confirm('Are you sure you want to reset all form values to defaults?')) {
        predictionForm.reset();
      }
    });
  }

  // 4. Auto-dismiss flash alerts after 5 seconds
  const autoAlerts = document.querySelectorAll('.alert-dismissible');
  autoAlerts.forEach(function (alert) {
    setTimeout(function () {
      try {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
      } catch (err) {
        // Ignored if bootstrap is unavailable
      }
    }, 6000);
  });
});
