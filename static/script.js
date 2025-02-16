document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission
    
    
    const fileInput = document.getElementById('fileInput');
    
    if (!fileInput.files.length) {
        Swal.fire('Error!', 'Please select a file before uploading.', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    // Show loading state
    Swal.fire({
        title: 'Processing...',
        text: 'Classifying your image...',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });

    fetch('/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        Swal.close();  // Close loading state
        if (data.error) {
            Swal.fire('Error!', data.error, 'error');
        } else {
            // Display results
            document.querySelector('.result-section').style.display = 'block';
            document.getElementById('result').textContent = `Prediction: ${data.prediction}`;
            document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            document.getElementById('catProb').textContent = `Cat Probability: ${(data.cat_probability * 100).toFixed(2)}%`;
            document.getElementById('dogProb').textContent = `Dog Probability: ${(data.dog_probability * 100).toFixed(2)}%`;

            // Display image
            const uploadedImage = document.getElementById('uploadedImage');
            uploadedImage.src = URL.createObjectURL(fileInput.files[0]);
        }
    })
    .catch(error => {
        Swal.close();
        Swal.fire('Error!', 'An error occurred while processing your image.', 'error');
    });
});
