document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("image-form");
    const fileInput = form.querySelector("input[name='file']");
    const spinner = document.getElementById("spinner");
    const buttonText = document.getElementById("button-text");
    const uploadBtn = form.querySelector(".upload-btn");
  
    fileInput.addEventListener("change", () => {
      console.log(fileInput.files.length ? fileInput.files[0].name : "No file chosen");
    });
  
    form.onsubmit = async (e) => {
      e.preventDefault();
  
      if (!fileInput.files.length) {
        alert("Please choose a file first");
        return;
      }
  
      const formData = new FormData();
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
  });
  