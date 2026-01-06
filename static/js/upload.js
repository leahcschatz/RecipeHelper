document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("url-form").addEventListener("submit", async function (e) {
      e.preventDefault();
      const urlInput = document.querySelector('input[name="url"]');
      const url = urlInput.value.trim();
      if (!url) return alert("Please enter a URL.");
    
      const formData = new FormData();
      formData.append("url", url);
      const button = this.querySelector("button");
      const buttonText = button.querySelector("span");
      buttonText.textContent = "Processing...";
      button.disabled = true;
    
      try {
        const response = await fetch("/process_url", {
          method: "POST",
          body: formData,
          redirect: 'follow'
        });
    
        if (!response.ok) throw new Error("Failed to process URL.");
    
        // Check if response was a redirect (fetch automatically followed it)
        if (response.redirected) {
          // Update the URL in the browser and navigate to the recipe page
          window.location.href = response.url;
          return;
        }
    
        // Render the recipe page HTML returned by Flask
        const html = await response.text();
        document.open();
        document.write(html);
        document.close();
    
      } catch (err) {
        console.error(err);
        alert("Error processing URL. Please check the link and try again.");
      } finally {
        buttonText.textContent = "Submit URL";
        button.disabled = false;
      }
    });
    
  });
  