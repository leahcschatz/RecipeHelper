document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("image-form");
    const fileInput = form.querySelector("input[name='file']");
    const spinner = document.getElementById("spinner");
    const buttonText = document.getElementById("button-text");
    const uploadBtn = form.querySelector(".upload-btn");

    fileInput.addEventListener("change", () => {
      const fileCount = fileInput.files.length;
      if (fileCount > 0) {
        console.log(`${fileCount} file(s) selected:`, Array.from(fileInput.files).map(f => f.name).join(", "));
      } else {
        console.log("No files chosen");
      }
    });
  
    form.onsubmit = async (e) => {
      e.preventDefault();
  
      if (!fileInput.files.length) {
        alert("Please choose at least one file first");
        return;
      }
  
      // Create new FormData for each submission
      const formData = new FormData();
      
      // Append all selected files
      for (let i = 0; i < fileInput.files.length; i++) {
        formData.append("file", fileInput.files[i]);
      }
  
      try {
        spinner.style.display = "inline-block";
        const fileCount = fileInput.files.length;
        buttonText.textContent = fileCount > 1 ? `Processing ${fileCount} files...` : "Uploading...";
        uploadBtn.disabled = true;
  
        const res = await fetch("/process_file", {
          method: "POST",
          body: formData,
        });
  
        spinner.style.display = "none";
        uploadBtn.disabled = false;
        buttonText.textContent = "Upload";
  
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({ error: res.statusText }));
          alert(`Upload failed: ${errorData.error || res.statusText}`);
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
  