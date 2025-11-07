document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("image-form");
    const fileInput = form.querySelector("input[name='file']");
    const spinner = document.getElementById("spinner");
    const buttonText = document.getElementById("button-text");
    const uploadBtn = form.querySelector(".upload-btn");
    const formData = new FormData();

    fileInput.addEventListener("change", () => {
      console.log(fileInput.files.length ? fileInput.files[0].name : "No file chosen");
    });
  
    form.onsubmit = async (e) => {
      e.preventDefault();
  
      if (!fileInput.files.length) {
        alert("Please choose a file first");
        return;
      }
  
      formData.append("file", fileInput.files[0]);
  
      try {
        spinner.style.display = "inline-block";
        buttonText.textContent = "Uploading...";
        uploadBtn.disabled = true;
  
        const res = await fetch("/process_file", {
          method: "POST",
          body: formData,
        });
  
        spinner.style.display = "none";
        uploadBtn.disabled = false;
        buttonText.textContent = "Upload";
  
        if (!res.ok) {
          alert(`Upload failed: ${res.statusText}`);
          return;
        }
  
        const html = await res.text();
        document.open();
        document.write(html);
        document.close();
      } catch (err) {
        spinner.style.display = "none";
        uploadBtn.disabled = false;
        buttonText.textContent = "Upload";
        alert(`Upload error: ${err}`);
      }
    };

    document.getElementById("url-form").addEventListener("submit", async function (e) {
      e.preventDefault();
      const urlInput = document.querySelector('input[name="url"]');
      const url = urlInput.value.trim();
      if (!url) return alert("Please enter a URL.");
    
      formData.append("url", url);
      const button = this.querySelector("button");
      const buttonText = button.querySelector("span");
      buttonText.textContent = "Processing...";
      button.disabled = true;
    
      try {
        const response = await fetch("/process_url", {
          method: "POST",
          body: formData,
        });
    
        if (!response.ok) throw new Error("Failed to process URL.");
    
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
  