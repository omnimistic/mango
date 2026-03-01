function generateImage() {
    const promptInput = document.getElementById('promptBox');
    const prompt = promptInput.value.trim();
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const statusText = document.getElementById('statusText');
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const generateBtn = document.getElementById('generateBtn');
    
    if (!prompt) return;

    // Initialize generation state
    generateBtn.disabled = true;
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    statusText.innerText = "Connecting to Engine...";

    // Establish SSE connection to inference backend
    const eventSource = new EventSource(`/generate_stream?prompt=${encodeURIComponent(prompt)}`);

    // Handle streaming progress updates
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.status) {
            statusText.innerText = data.status;
        } 
        else if (data.step) {
            // Calculate and render progress percentage
            const percentage = (data.step / data.total) * 100;
            progressFill.style.width = percentage + '%';
            statusText.innerText = `[${data.vibe} MODE] Denoising Step: ${data.step} / ${data.total}`;
        } 
        else if (data.image_url) {
            // Generation complete: cleanup and render result
            eventSource.close();
            
            statusText.innerText = "Complete!";
            setTimeout(() => {
                progressContainer.style.display = 'none';
                statusText.innerText = "";
                generateBtn.disabled = false;
                
                modalImg.src = `${data.image_url}?t=${new Date().getTime()}`;
                modal.style.display = 'block';
            }, 500);
        }
    };

    // Handle connection or inference errors
    eventSource.onerror = function(err) {
        console.error("EventSource failed:", err);
        eventSource.close();
        statusText.innerText = "Engine Error. Check Terminal.";
        generateBtn.disabled = false;
    };
}

function closeModal() {
    document.getElementById('imageModal').style.display = "none";
}

document.getElementById('promptBox').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') generateImage();
});