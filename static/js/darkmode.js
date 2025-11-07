document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const label = document.querySelector(".toggle-label");
  
    if (themeToggle && label) {
      // Check for saved theme preference or default to system preference
      const savedTheme = localStorage.getItem("theme");
      const systemPrefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      
      // Determine initial theme: saved preference > system preference > light
      let initialTheme = savedTheme;
      if (!initialTheme) {
        initialTheme = systemPrefersDark ? "dark" : "light";
      }
      
      // Apply initial theme
      if (initialTheme === "dark") {
        document.body.classList.add("dark-mode");
        themeToggle.checked = true;
        label.textContent = "Light Mode";
      } else {
        document.body.classList.remove("dark-mode");
        themeToggle.checked = false;
        label.textContent = "Dark Mode";
      }
      
      // Listen for system theme changes (if user hasn't manually set a preference)
      if (window.matchMedia) {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        mediaQuery.addEventListener("change", (e) => {
          // Only auto-update if user hasn't manually set a preference
          if (!savedTheme) {
            if (e.matches) {
              document.body.classList.add("dark-mode");
              themeToggle.checked = true;
              label.textContent = "Light Mode";
            } else {
              document.body.classList.remove("dark-mode");
              themeToggle.checked = false;
              label.textContent = "Dark Mode";
            }
          }
        });
      }
  
      // Handle manual toggle
      themeToggle.addEventListener("change", () => {
        if (themeToggle.checked) {
          document.body.classList.add("dark-mode");
          localStorage.setItem("theme", "dark");
          label.textContent = "Light Mode";
        } else {
          document.body.classList.remove("dark-mode");
          localStorage.setItem("theme", "light");
          label.textContent = "Dark Mode";
        }
      });
    }
  });
  