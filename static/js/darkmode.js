document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const label = document.querySelector(".toggle-label");
  
    if (themeToggle && label) {
      const currentTheme = localStorage.getItem("theme");
      if (currentTheme === "dark") {
        document.body.classList.add("dark-mode");
        themeToggle.checked = true;
        label.textContent = "Light Mode";
      }
  
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
  